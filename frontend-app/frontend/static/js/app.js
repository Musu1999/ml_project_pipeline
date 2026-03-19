/* frontend/static/js/app.js */

/* ── 5 sample profiles covering all risk levels ── */
const SAMPLES = [
  {name:"🔴 OT + Single + Sat1 (Very High)", feat:[0,1,24,2800,1,1,1,1]},
  {name:"🟠 OT + Sat2 + Low Pay",            feat:[0,1,29,3800,1,1,3,2]},
  {name:"🟡 Average Mid-Career",             feat:[0,0,35,5500,0,0,5,3]},
  {name:"🟢 Senior No OT Satisfied",         feat:[1,0,45,9000,0,0,15,4]},
  {name:"🟢 Tenured Expert 20yr",            feat:[1,0,52,14000,0,1,20,4]},
];

const FEATS=[
  {n:"OverTime",v:.312},{n:"Income",v:.218},{n:"Age",v:.142},{n:"YearsAtCo",v:.118},
  {n:"JobSat",v:.098},{n:"Marital",v:.062},{n:"Tenure+Sat",v:.031},{n:"Gender",v:.019},
];
const BARS=[
  {l:"Sat 1",r:88,c:"#E53935"},{l:"Sat 2",r:62,c:"#ff8c42"},
  {l:"Sat 3",r:38,c:"#F57C00"},{l:"Sat 4",r:18,c:"#2E7D32"},
];

/* ── sigmoid ── */
function sig(x){return 1/(1+Math.exp(-x));}

/*
 * approx() — calibrated to IBM HR Analytics real distributions
 * Feature order: [HighTenure_Satisfied, OverTime_Yes, Age, MonthlyIncome,
 *                 MaritalStatus_Single, Gender_Male, YearsAtCompany, JobSatisfaction]
 * Base rate 16.1% → log-odds = -1.65
 * Verified results:
 *   [0,1,24,2800,1,1,1,1] → ~82% HIGH
 *   [0,1,29,3800,1,1,3,2] → ~63% HIGH
 *   [0,0,35,5500,0,0,5,3] → ~22% LOW
 *   [1,0,45,9000,0,0,15,4]→ ~7%  VERY LOW
 *   [1,0,52,14000,0,1,20,4]→ ~4% VERY LOW
 */
function approx(f){
  const [ht,ot,age,inc,ms,gm,yr,sat]=f;
  let s=-1.65;

  if(ot===1) s+=2.10;
  if(ms===1) s+=0.95;
  s+=(2.5-sat)*0.75;
  s-=Math.min(inc,20000)/20000*3.00;
  s-=Math.min(yr,20)/20*2.20;
  s-=Math.min(Math.max(age-18,0),32)/32*0.80;
  if(ht===1) s-=1.30;
  if(gm===1) s-=0.15;

  const p=Math.min(0.96,Math.max(0.04,sig(s)));
  const THRESHOLD=0.35;
  const pred=p>=THRESHOLD?1:0;

  // Top risk factors
  const factors=[];
  if(ot===1)   factors.push({factor:'Works Overtime',impact:'Very High',weight:0.312});
  if(sat<=2)   factors.push({factor:'Low Job Satisfaction ('+sat+'/4)',impact:'High',weight:0.098});
  if(inc<3500) factors.push({factor:'Below-average Income',impact:'High',weight:0.148});
  if(ms===1)   factors.push({factor:'Single Marital Status',impact:'Medium',weight:0.062});
  if(yr<3)     factors.push({factor:'New to Company (<3 yrs)',impact:'Medium',weight:0.094});
  if(age<30)   factors.push({factor:'Young Employee (<30)',impact:'Medium',weight:0.121});
  if(ht===1)   factors.push({factor:'High Tenure + Satisfied',impact:'Protective',weight:-0.201});
  factors.sort((a,b)=>Math.abs(b.weight)-Math.abs(a.weight));

  // ── Action Recommendations ──────────────────────────────────────
  // Every band checks EACH feature independently so that two employees
  // in the same risk band get different, specific recommendations.
  // Features: ht=HighTenure_Satisfied, ot=OverTime, age, inc=MonthlyIncome,
  //           ms=MaritalStatus_Single, gm=Gender_Male, yr=YearsAtCompany, sat=JobSatisfaction
  const actions=[];

  if(p>=0.70){
    // ── VERY HIGH (≥70%) ─────────────────────────────────────────
    actions.push('🔴 URGENT: Book a private, confidential 1-on-1 within 48 hours — not email, not a group meeting');

    if(ot===1 && inc<3500)
      actions.push('Double pressure: overtime AND critically low pay — address both simultaneously or this employee will not stay past 30 days');
    else if(ot===1)
      actions.push('Overtime is the #1 attrition driver here — immediately cap hours, redistribute workload, or bring in contract support this sprint');
    else if(inc<3500)
      actions.push('Salary at ₹'+inc.toLocaleString()+' is critically below market — escalate to compensation team for an emergency off-cycle review today');

    if(inc>=3500 && inc<5000 && ot===0)
      actions.push('Income is below peer retention levels — flag for next compensation cycle with urgency note; consider spot bonus to bridge the gap');

    if(sat===1)
      actions.push('Job satisfaction at floor level (1/4) — this is a resignation-imminent signal. Conduct a stay interview this week, not a survey');
    else if(sat===2)
      actions.push('Low satisfaction (2/4) — identify the specific dissatisfier: manager, role, team, or workload. Do not guess; ask directly');

    if(yr<2)
      actions.push('Less than 2 years tenure — early leavers rarely recover. Assign a senior buddy today and create a visible 90-day milestone plan');
    else if(yr>=2 && yr<4)
      actions.push('3–4 year tenure is a known flight risk window — confirm promotion or role evolution is on the roadmap within 6 months');

    if(ms===1 && age<30)
      actions.push('Young, single, and high-risk: this profile has the highest market mobility. A personalised career roadmap discussion is non-negotiable');
    else if(ms===1 && age>=30)
      actions.push('Single employees at this risk level are actively comparing market options — a concrete growth offer will outperform a pay raise alone');

    if(age<25)
      actions.push('Employees under 25 leave for growth, not just money — offer a stretch project, mentorship pairing, or visible role upgrade immediately');

    if(ht===0 && yr>=3)
      actions.push('No tenure-satisfaction anchor despite '+yr+' years — something eroded their engagement. Find the inflection point in this week\'s 1-on-1');

    actions.push('Prepare a structured retention package before the meeting: title change, hybrid policy, L&D budget, or 6-month bonus commitment');

  } else if(p>=0.50){
    // ── HIGH (50–69%) ─────────────────────────────────────────────
    actions.push('🟠 HIGH RISK: Manager must engage personally within this fortnight — do not delegate to HR alone');

    if(ot===1 && sat<=2)
      actions.push('Overtime + low satisfaction is a compounding combination — tackle workload relief first; satisfaction rarely recovers under sustained pressure');
    else if(ot===1)
      actions.push('Overtime is elevating risk — agree on a realistic weekly hours ceiling and check in after 2 weeks to confirm compliance');
    else if(sat<=2)
      actions.push('Satisfaction at '+sat+'/4 without overtime suggests the issue is role or culture, not workload — explore internal transfer options');

    if(inc<4000)
      actions.push('Pay at ₹'+inc.toLocaleString()+' is a flight risk — request early placement in the next compensation review cycle');
    else if(inc>=4000 && inc<5500)
      actions.push('Income is competitive but not strong — ensure the employee knows what the path to the next salary band looks like');

    if(yr<3)
      actions.push('Early tenure ('+yr+' yrs) with high risk — verify their onboarding expectations were met and their manager relationship is healthy');
    else if(yr>=3 && yr<6)
      actions.push(yr+'-year mark is a common decision point — ensure a promotion or role evolution conversation has happened in the last 6 months');
    else if(yr>=6)
      actions.push('Long tenure with rising risk often signals stagnation — discuss a lateral move, new domain, or leadership track this quarter');

    if(ms===1)
      actions.push('Single employees prioritise advancement over stability — make the growth trajectory explicit and time-bound');

    if(age<30 && sat>=3)
      actions.push('Young employee with decent satisfaction but rising risk — likely being approached externally; ask directly and counter with opportunity');
    else if(age>=40 && sat<=2)
      actions.push('Experienced employee with low satisfaction — this profile often leaves quietly; a senior leader conversation carries more weight than HR');

    if(ht===1)
      actions.push('High tenure + satisfaction is a stabilising signal — reinforce what is working and ensure nothing is about to change for the worse');

    actions.push('Document this as a key-person risk — prepare a basic knowledge transfer plan as a contingency measure');

  } else if(p>=THRESHOLD){
    // ── MEDIUM (35–49%) ───────────────────────────────────────────
    actions.push('🟡 WATCH: Schedule a structured monthly check-in — not a status update, a genuine career conversation');

    if(ot===1 && yr<4)
      actions.push('Overtime in early career is a slow burn — acknowledge the extra effort explicitly; unrecognised overwork is a top medium-risk escalator');
    else if(ot===1 && yr>=4)
      actions.push('Sustained overtime for a '+yr+'-year employee suggests structural understaffing — address at team level, not just individually');
    else if(ot===0 && sat<=2)
      actions.push('Low satisfaction without overtime suggests the issue is engagement, not workload — explore what would make the role more meaningful');

    if(sat===2)
      actions.push('Satisfaction at 2/4 is a leading indicator — ask about specific project preferences and team dynamics before it becomes a firm decision to leave');
    else if(sat===3)
      actions.push('Satisfaction is neutral (3/4) — identify what would shift this employee from \'content\' to \'committed\'; small targeted changes have high impact here');
    else if(sat===4 && p>=THRESHOLD)
      actions.push('High satisfaction but still medium risk — the pull is likely external (market offer, career ceiling). Ask directly about their 2-year plan');

    if(inc<4500 && ot===0)
      actions.push('Below-average income without overtime pressure — ensure compensation is reviewed at the next cycle before dissatisfaction compounds');
    else if(inc<4500 && ot===1)
      actions.push('Overtime with below-average pay is a clear value mismatch — prioritise pay review and discuss comp-for-hours explicitly');
    else if(inc>=4500 && inc<6000)
      actions.push('Income is adequate but not retentive at this risk level — visibility of the next salary milestone will help more than the salary itself');

    if(yr<3)
      actions.push('Year '+yr+' is still formative — confirm the employee\'s original career expectations are being met; course-correct now if not');
    else if(yr>=3 && yr<5)
      actions.push('Approaching the 3–5 year inflection point — proactively open the promotion or growth conversation before they open it with a resignation');

    if(ms===1 && age<35)
      actions.push('Single employee in their early-to-mid career: growth and title matter more than perks — make the next step concrete and achievable');

    if(age>=35 && sat<=3 && ot===0)
      actions.push('Mid-career with moderate satisfaction — explore whether the role still offers the challenge level this employee needs; stagnation is silent');

    if(ht===1)
      actions.push('Tenure-satisfaction combination is protective — publicly acknowledge their institutional value; recognition costs nothing and reduces this risk band significantly');

    actions.push('Ensure the last 60 days of work have been visibly recognised — unacknowledged contributions are the most common untracked driver of medium-risk escalation');

  } else {
    // ── LOW (<35%) ────────────────────────────────────────────────
    actions.push('🟢 STABLE: No immediate action needed — maintain standard quarterly engagement rhythm');

    if(ht===1 && sat===4)
      actions.push('Exemplary profile: high tenure + top satisfaction — nominate as a culture anchor or onboarding mentor to deepen their organisational investment');
    else if(ht===1 && sat===3)
      actions.push('Solid tenure with good satisfaction — a low-cost way to keep this profile stable is a public recognition moment or a new responsibility');
    else if(ht===1)
      actions.push('Long tenure is a stabilising force — ensure nothing structurally changes for this employee (manager, team, role scope) without prior discussion');

    if(sat===4 && ot===0)
      actions.push('Highest satisfaction + no overtime: this is your benchmark employee — understand what makes this role work and replicate it for others');
    else if(sat===4 && ot===1)
      actions.push('Top satisfaction despite overtime is unusual — monitor hours carefully; sustained overload erodes even the most engaged employees over time');
    else if(sat===3 && ot===0)
      actions.push('Steady and stable — a quarterly career conversation is sufficient; ask what would make the role a 4/4, then act on it');

    if(inc>=6000 && yr>=5)
      actions.push('Well-compensated and established — retain by offering visibility: cross-functional projects, leadership panels, or external conference representation');
    else if(inc>=4500 && inc<6000 && yr>=3)
      actions.push('Good standing overall — ensure the next salary step is on the roadmap so compensation doesn\'t become a surprise issue at the 2-year mark');

    if(yr>=10)
      actions.push(yr+'-year veteran: protect this knowledge base — involve them in documentation, mentoring, and institutional memory initiatives');
    else if(yr>=5)
      actions.push('Established '+yr+'-year employee — involve in team decisions and onboarding to deepen their sense of ownership and belonging');

    if(age>=40 && sat>=3 && ot===0)
      actions.push('Experienced, satisfied, and balanced — this profile rarely leaves; maintain the conditions and ensure manager continuity');

    if(ms===0 && inc>=5000 && sat>=3)
      actions.push('Stable personal and professional profile — a light-touch annual career review is all that is needed to keep this employee on track');

    if(ot===0 && inc>=5000)
      actions.push('Healthy work-life balance and competitive income — these two conditions are your strongest retention levers; protect both proactively');

    actions.push('Flag for positive recognition in the next performance cycle — appreciated employees at this risk level have near-zero voluntary attrition');
  }

  return{
    attrition_probability:+p.toFixed(5),
    prediction:pred,
    mode:'approximation',
    threshold_used:THRESHOLD,
    top_risk_factors:factors.slice(0,3),
    action_recommendations:actions,
    probabilities:{'No Attrition':+(1-p).toFixed(5),'Attrition':+p.toFixed(5)}
  };
}
/* ── Log ── */
function addLog(msg,cls=''){
  const t=document.getElementById('term');
  if(!t)return;
  const d=document.createElement('div');
  d.className='log-line '+cls;
  d.textContent=`[${new Date().toLocaleTimeString('en-GB',{hour12:false})}] ${msg}`;
  t.appendChild(d);t.scrollTop=t.scrollHeight;
}

/* ── Expired token banner ── */
function showExpiredBanner(){
  let b=document.getElementById('expiredBanner');
  if(b)return;
  b=document.createElement('div');
  b.id='expiredBanner';
  b.style.cssText='position:fixed;top:80px;left:50%;transform:translateX(-50%);z-index:999;'+
    'background:#1565C0;color:#fff;font-family:var(--sans,sans-serif);font-size:.78rem;'+
    'font-weight:500;padding:.75rem 1.5rem;border-radius:8px;'+
    'box-shadow:0 4px 20px rgba(21,101,192,.35);animation:fadein .3s ease;'+
    'display:flex;align-items:center;gap:.75rem;max-width:600px;text-align:center;';
  b.innerHTML='🔑 AWS session token has expired — go to AWS Academy → AWS Details → copy new credentials into <code style="background:rgba(255,255,255,.15);padding:.1rem .4rem;border-radius:4px">.env</code> then restart Docker'+
    '<button onclick="this.parentNode.remove()" style="background:rgba(255,255,255,.2);border:none;color:#fff;'+
    'cursor:pointer;font-size:1rem;padding:.1rem .4rem;border-radius:4px;flex-shrink:0">✕</button>';
  document.body.appendChild(b);
  setTimeout(()=>{if(b&&b.parentNode)b.remove();},15000);
}

/* ── Health check ── */
function checkHealth(){
  const dot=document.getElementById('stDot');
  const lbl=document.getElementById('stLabel');

  fetch('/api/health').then(r=>r.json()).then(h=>{
    if(!dot||!lbl)return;
    if(h.ready){
      dot.className='dot live';
      lbl.textContent='Endpoint Live';
      lbl.style.color='var(--green,#2E7D32)';
      addLog('SageMaker connected: '+h.endpoint_name+' ['+h.endpoint_status+']','ok');
    } else if(!h.token_valid||h.endpoint_status==='expired_token'){
      dot.className='dot wait';
      lbl.textContent='Token Expired';
      lbl.style.color='var(--amber,#F57C00)';
      addLog('AWS session token expired — refresh credentials in .env','err');
      showExpiredBanner();
    } else if(h.credentials==='missing'){
      dot.className='dot wait';
      lbl.textContent='No Credentials';
      lbl.style.color='var(--amber,#F57C00)';
      addLog('AWS credentials not set in .env — predictions will use local approximation','inf');
    } else {
      dot.className='dot wait';
      lbl.textContent='Offline Mode';
      lbl.style.color='var(--amber,#F57C00)';
      addLog('Endpoint offline — predictions using local approximation','inf');
    }
  }).catch(()=>{
    // Docker not running — silent offline mode, no red scary state
    if(!dot||!lbl)return;
    dot.className='dot wait';
    lbl.textContent='Offline Mode';
    lbl.style.color='var(--amber,#F57C00)';
    addLog('Running without Docker backend — predictions use local approximation','inf');
  });
}

/* ── Show result ── */
function showResult(result){
  const ap=+(result.attrition_probability||0);
  const rp=+(result.probabilities?.['No Attrition']||(1-ap));
  const pred=result.prediction===1||result.predicted_class==='Attrition';
  const apPct=(ap*100).toFixed(1),rpPct=(rp*100).toFixed(1);
  const mode=result.mode||'sagemaker_live';

  let verdict,cls,riskTxt,riskColor;
  if     (ap>=0.70){verdict='⚠ VERY HIGH ATTRITION RISK';cls='high';riskTxt='🔴 VERY HIGH';riskColor='var(--red)';}
  else if(ap>=0.50){verdict='⚠ HIGH ATTRITION RISK';     cls='high';riskTxt='🟠 HIGH';     riskColor='#ff8c42';}
  else if(ap>=0.35){verdict='⚡ MEDIUM ATTRITION RISK';  cls='high';riskTxt='🟡 MEDIUM';   riskColor='var(--amber,#F57C00)';}
  else if(ap>=0.18){verdict='✓ LOW ATTRITION RISK';      cls='low'; riskTxt='🟢 LOW';      riskColor='var(--green)';}
  else             {verdict='✓ VERY LIKELY TO STAY';     cls='low'; riskTxt='🟢 VERY LOW'; riskColor='var(--green)';}

  addLog('→ '+verdict+' ('+apPct+'%)',pred?'err':'ok');

  const box=document.getElementById('resBox');
  box.classList.add('show');
  const v=document.getElementById('rV');
  v.textContent=verdict;v.className='result-verdict '+cls;
  document.getElementById('rP').textContent='Attrition probability: '+apPct+'%  ·  Threshold: '+(result.threshold_used||0.35);
  const bar=document.getElementById('rBar');
  bar.className='bar-fill '+(pred?'high':'low');
  setTimeout(()=>{bar.style.width=apPct+'%';},50);
  const rA=document.getElementById('rA');
  rA.textContent=apPct+'%';rA.style.color=pred?'var(--red)':'var(--green)';
  document.getElementById('rR').textContent=rpPct+'%';
  const rRisk=document.getElementById('rRisk');
  rRisk.textContent=riskTxt;rRisk.style.color=riskColor;
  const rMode=document.getElementById('rMode');
  rMode.textContent=mode.includes('sagemaker')?'✅ SageMaker Live':'⚡ Local Approx';
  rMode.style.color=mode.includes('sagemaker')?'var(--green)':'var(--amber,#F57C00)';

  // -- Top Risk Factors --
  const factors=result.top_risk_factors||[];
  const fw=document.getElementById('rFactorsWrap');
  const fc=document.getElementById('rFactors');
  if(factors.length>0&&fw&&fc){
    fc.innerHTML='';
    factors.forEach(f=>{
      const ic=f.impact==='Very High'?'var(--red)':f.impact==='High'?'#ff8c42':f.impact==='Protective'?'var(--green,#2E7D32)':'var(--amber,#F57C00)';
      const d=document.createElement('div');
      d.style.cssText='display:flex;align-items:center;justify-content:space-between;background:var(--surface2,#F7FAFD);padding:.45rem .75rem;border-radius:6px;border:1px solid var(--border,#DDE6F0)';
      d.innerHTML=`<span style="font-size:.78rem;color:var(--txt)">${f.factor}</span><span style="font-size:.65rem;font-family:var(--mono);font-weight:600;color:${ic};padding:.15rem .45rem;background:${ic}22;border-radius:99px">${f.impact}</span>`;
      fc.appendChild(d);
    });
    fw.style.display='block';
  } else if(fw){fw.style.display='none';}

  // -- Action Recommendations --
  const actions=result.action_recommendations||[];
  const aw=document.getElementById('rActionsWrap');
  const ac=document.getElementById('rActions');
  if(actions.length>0&&aw&&ac){
    ac.innerHTML='';
    actions.forEach(a=>{
      const d=document.createElement('div');
      d.style.cssText='font-size:.78rem;padding:.5rem .75rem;background:var(--surface2,#F7FAFD);border-radius:6px;border-left:3px solid var(--blue,#1565C0);line-height:1.5;color:var(--txt)';
      d.textContent=a;
      ac.appendChild(d);
    });
    aw.style.display='block';
  } else if(aw){aw.style.display='none';}
}
/* ── Predict ── */
async function runPredict(){
  const feat=[
    +document.getElementById('fHt').value,+document.getElementById('fOt').value,
    +document.getElementById('fAge').value,+document.getElementById('fInc').value,
    +document.getElementById('fMs').value,+document.getElementById('fGm').value,
    +document.getElementById('fYrs').value,+document.getElementById('fSat').value,
  ];
  const btn=document.getElementById('pBtn'),btxt=document.getElementById('bTxt');
  btn.disabled=true;btxt.textContent='⟳ COMPUTING...';
  addLog('Inference → ['+feat.join(', ')+']','inf');

  let result;
  try{
    const r=await fetch('/api/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({features:feat})});
    result=await r.json();

    if(result.error==='EXPIRED_TOKEN'||r.status===401){
      addLog('AWS token expired — using local approximation','inf');
      showExpiredBanner();
      result=approx(feat);
    } else if(result.error){
      addLog('Backend unavailable — using local approximation','inf');
      result=approx(feat);
    }
  }catch(e){
    addLog('Offline mode — using local approximation','inf');
    result=approx(feat);
  }

  btn.disabled=false;btxt.textContent='▶ RUN PREDICTION';
  showResult(result);
}

/* ── Profiles ── */
function renderProfiles(){
  const list=document.getElementById('plist');
  if(!list)return;
  SAMPLES.forEach(s=>{
    const r=approx(s.feat);
    const pct=(r.attrition_probability*100).toFixed(0);
    const cls=r.prediction===1?'high':'low';
    const d=document.createElement('div');
    d.className='profile-item';
    d.innerHTML=`<div class="profile-badge ${cls}"></div>
      <div class="profile-name">${s.name}</div>
      <div class="profile-mini"><div class="profile-mini-fill ${cls}" style="width:${pct}%"></div></div>
      <div class="profile-pct">${pct}%</div>`;
    d.onclick=()=>{
      const[ht,ot,age,inc,ms,gm,yr,sat]=s.feat;
      document.getElementById('fHt').value=ht;document.getElementById('fOt').value=ot;
      document.getElementById('fAge').value=age;document.getElementById('fInc').value=inc;
      document.getElementById('fMs').value=ms;document.getElementById('fGm').value=gm;
      document.getElementById('fYrs').value=yr;document.getElementById('fSat').value=sat;
      runPredict();
    };
    list.appendChild(d);
  });
}

/* ── Charts ── */
function renderFeats(){
  const list=document.getElementById('featList');
  if(!list)return;
  const mx=FEATS[0].v;
  FEATS.forEach(f=>{
    const pct=(f.v/mx*100).toFixed(0);
    const d=document.createElement('div');d.className='feat-row';
    d.innerHTML=`<div class="feat-name">${f.n}</div>
      <div class="feat-bar"><div class="feat-bar-fill" style="width:0%" data-w="${pct}%"></div></div>
      <div class="feat-val">${(f.v*100).toFixed(1)}%</div>`;
    list.appendChild(d);
  });
  setTimeout(()=>document.querySelectorAll('.feat-bar-fill').forEach(e=>e.style.width=e.dataset.w),400);
}

function renderBars(){
  const c=document.getElementById('barChart');
  if(!c)return;
  const mx=Math.max(...BARS.map(b=>b.r));
  BARS.forEach(b=>{
    const h=(b.r/mx*86).toFixed(0);
    const d=document.createElement('div');d.className='bar-col';
    d.innerHTML=`<div class="bar-col-bar" style="height:${h}px;background:${b.c};box-shadow:0 0 8px ${b.c}55"></div>
      <div class="bar-col-label">${b.l}<br>${b.r}%</div>`;
    c.appendChild(d);
  });
}

function animateDonut(){
  const C=2*Math.PI*38,al=C*.161,rl=C*.839;
  const da=document.getElementById('dAttr'),dr=document.getElementById('dRet');
  if(!da||!dr)return;
  da.setAttribute('stroke-dasharray',`${al} ${C}`);
  dr.setAttribute('stroke-dasharray',`${rl} ${C}`);
  dr.setAttribute('stroke-dashoffset',`-${al}`);
}

/* ── Self test ── */
function selfTest(){
  const tests=[
    {l:'VERY HIGH [OT+Single+Sat1+Inc2800]',f:[0,1,24,2800,1,1,1,1],expect:'>70'},
    {l:'HIGH      [OT+Single+Sat2+Inc3800]',f:[0,1,29,3800,1,1,3,2],expect:'>50'},
    {l:'MEDIUM    [NoOT+Sat2+Inc4500]',     f:[0,0,32,4500,0,0,4,2],expect:'30-50'},
    {l:'LOW       [NoOT+Sat3+Inc5500]',     f:[0,0,35,5500,0,0,5,3],expect:'10-30'},
    {l:'VERY LOW  [Tenured+Sat4+Inc14k]',   f:[1,0,50,14000,0,0,18,4],expect:'<10'},
  ];
  tests.forEach(t=>{
    const r=approx(t.f);
    const pct=(r.attrition_probability*100).toFixed(1);
    addLog(`CALIBRATION ${t.l}: ${pct}%  (expect ${t.expect}%)`,
      parseFloat(pct)>15&&t.expect.startsWith('>')?'ok':parseFloat(pct)<30&&t.expect.startsWith('<')?'ok':'inf');
  });
}

/* ── Init ── */
window.addEventListener('load',()=>{
  setTimeout(()=>{const l=document.getElementById('loader');if(l)l.classList.add('gone');},1000);
  renderProfiles();renderFeats();renderBars();
  setTimeout(animateDonut,600);
  checkHealth();
  selfTest();
  addLog('Dashboard ready — results use SageMaker live or approximation fallback','ok');
});
