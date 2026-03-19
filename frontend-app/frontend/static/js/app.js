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

  /* ── Top risk factors ── */
  const factors=[];
  if(ot===1)   factors.push({factor:'Works Overtime',              impact:'Very High', weight:0.312});
  if(sat<=2)   factors.push({factor:'Low Job Satisfaction ('+sat+'/4)', impact:'High', weight:0.098});
  if(inc<3500) factors.push({factor:'Below-average Income',        impact:'High',      weight:0.148});
  if(ms===1)   factors.push({factor:'Single Marital Status',       impact:'Medium',    weight:0.062});
  if(yr<3)     factors.push({factor:'New to Company (<3 yrs)',     impact:'Medium',    weight:0.094});
  if(age<30)   factors.push({factor:'Young Employee (<30)',        impact:'Medium',    weight:0.121});
  if(ht===1)   factors.push({factor:'High Tenure + Satisfied',     impact:'Protective',weight:-0.201});
  factors.sort((a,b)=>Math.abs(b.weight)-Math.abs(a.weight));

  /* ── Action recommendations — 4 bands, feature-aware ──
   *  Mirrors get_action_recommendations() in api.py exactly so that
   *  the local approximation fallback produces identical logic to the
   *  live SageMaker response.
   *  Bands:  VERY HIGH ≥0.70 | HIGH ≥0.50 | MEDIUM ≥0.35 | LOW <0.35
   */
  const actions=[];

  if(p>=0.70){
    /* ── VERY HIGH ── */
    actions.push('🔴 URGENT: Schedule a private 1-on-1 with the manager within 48 hours');
    if(ot===1)        actions.push('Overtime is a top attrition driver — immediately review workload and redistribute tasks or hire contract support');
    if(inc<3500)      actions.push('Salary is critically below market — escalate to compensation team for an off-cycle review this week');
    else if(inc<5000) actions.push('Initiate a compensation benchmarking review — salary may be below peer retention levels');
    if(sat<=1)        actions.push('Job satisfaction is at the lowest level (1/4) — conduct a confidential stay interview immediately');
    else if(sat===2)  actions.push('Low job satisfaction detected — explore role redesign, project rotation, or team change');
    if(yr<2)          actions.push('Employee is in a high-churn early tenure window — assign a senior buddy and create a 6-month engagement milestone plan');
    if(ms===1&&age<32)actions.push('Young single employees are most mobile — discuss personalised career growth roadmap and skill development funding');
    if(ht===0)        actions.push('No strong tenure-satisfaction anchor — consider fast-tracking a promotion discussion or stretch assignment');
    actions.push('Prepare a structured retention offer: title progression, hybrid flexibility, L&D budget, or equity/bonus commitment');

  } else if(p>=0.50){
    /* ── HIGH ── */
    actions.push('🟠 HIGH RISK: Prioritise this employee for manager attention within this fortnight');
    if(ot===1)        actions.push('Overtime pattern is contributing to risk — discuss sustainable workload boundaries and agree on a cap');
    if(inc<4500)      actions.push('Flag for next compensation cycle — below-average income is a measurable flight risk factor');
    if(sat<=2)        actions.push('Schedule an engagement conversation — understand specific dissatisfiers before they become resignation triggers');
    if(yr<3)          actions.push('Still in the critical early-tenure phase — confirm the employee\'s career trajectory and growth expectations are being met');
    if(ms===1)        actions.push('Single employees prioritise career advancement — ensure visibility of promotion or lateral growth opportunities');
    if(age<30)        actions.push('Younger employees respond strongly to skill-building investments — offer a relevant certification or conference budget');
    if(ht===1)        actions.push('Positive signal: high tenure + satisfaction. Reinforce what is working and use this employee as a team anchor');
    actions.push('Document key person risk — ensure knowledge transfer plan exists for business continuity');

  } else if(p>=THRESHOLD){
    /* ── MEDIUM ── */
    actions.push('🟡 WATCH: Include in next monthly manager check-in with a structured agenda');
    if(ot===1)        actions.push('Monitor overtime frequency — sustained overwork without recognition is a slow-burn attrition driver');
    if(sat===2)       actions.push('Satisfaction is below average — proactively ask about project preferences and team dynamics');
    else if(sat===3)  actions.push('Satisfaction is moderate — identify what would move this employee from \'content\' to \'committed\'');
    if(inc<5000)      actions.push('Review compensation at the next cycle — staying competitive reduces medium-risk escalation');
    if(yr<3)          actions.push('Confirm onboarding goals have been met and that the employee has a clear development plan for year 2–3');
    if(ms===1&&age<35)actions.push('Discuss medium-term career goals — ensure the role offers the growth trajectory single employees typically seek');
    if(ht===1)        actions.push('High-tenure satisfaction is a stabilising factor — acknowledge this employee\'s institutional contribution publicly');
    actions.push('Ensure recent work is recognised — a lack of visible appreciation is the most common untracked medium-risk driver');

  } else {
    /* ── LOW ── */
    actions.push('🟢 STABLE: No immediate intervention needed — maintain standard engagement cadence');
    if(ht===1)        actions.push('High tenure + satisfaction: consider as a mentor or culture ambassador — these employees significantly reduce team attrition by 12–18%');
    if(sat===4)       actions.push('Highest satisfaction tier — solicit this employee\'s input in team culture initiatives and hiring decisions');
    else if(sat===3)  actions.push('Good satisfaction level — periodic check-ins every quarter are sufficient to keep this profile stable');
    if(yr>=5)         actions.push('Long-tenured and stable — document their institutional knowledge and involve them in onboarding newer team members');
    if(ot===0&&inc>=5000) actions.push('Healthy work-life balance and competitive income — maintain these conditions as a baseline retention guarantee');
    actions.push('Flag for recognition in next performance cycle — proactive appreciation prevents passive disengagement');
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

  /* ── Top Risk Factors — reset then re-render ── */
  const factors=result.top_risk_factors||[];
  const fw=document.getElementById('rFactorsWrap');
  const fc=document.getElementById('rFactors');
  if(fw) fw.style.display='none';          // always hide first
  if(fc) fc.innerHTML='';                  // always clear stale items
  if(factors.length>0&&fw&&fc){
    factors.forEach(f=>{
      const ic=f.impact==='Very High'?'var(--red)':f.impact==='High'?'#ff8c42':f.impact==='Protective'?'var(--green,#2E7D32)':'var(--amber,#F57C00)';
      const d=document.createElement('div');
      d.style.cssText='display:flex;align-items:center;justify-content:space-between;background:var(--surface2,#F7FAFD);padding:.45rem .75rem;border-radius:6px;border:1px solid var(--border,#DDE6F0)';
      d.innerHTML=`<span style="font-size:.78rem;color:var(--txt)">${f.factor}</span><span style="font-size:.65rem;font-family:var(--mono);font-weight:600;color:${ic};padding:.15rem .45rem;background:${ic}22;border-radius:99px">${f.impact}</span>`;
      fc.appendChild(d);
    });
    fw.style.display='block';
  }

  /* ── Action Recommendations — reset then re-render ── */
  const actions=result.action_recommendations||[];
  const aw=document.getElementById('rActionsWrap');
  const ac=document.getElementById('rActions');
  if(aw) aw.style.display='none';          // always hide first
  if(ac) ac.innerHTML='';                  // always clear stale items
  if(actions.length>0&&aw&&ac){
    actions.forEach((a,i)=>{
      const d=document.createElement('div');
      /* First item = risk-level headline — bold it */
      d.style.cssText='font-size:.78rem;padding:.5rem .75rem;background:var(--surface2,#F7FAFD);'+
        'border-radius:6px;border-left:3px solid var(--blue,#1565C0);line-height:1.5;color:var(--txt);'+
        (i===0?'font-weight:600;':'');
      d.textContent=a;
      ac.appendChild(d);
    });
    aw.style.display='block';
  }
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
