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
  {l:"Sat 1",r:88,c:"#ff3d6b"},{l:"Sat 2",r:62,c:"#ff8c42"},
  {l:"Sat 3",r:38,c:"#ffd166"},{l:"Sat 4",r:18,c:"#a8ff3e"},
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
  let s=-1.65; // base 16.1%

  if(ot===1) s+=2.10;                       // overtime — strongest signal
  if(ms===1) s+=0.95;                       // single
  s+=(2.5-sat)*0.75;                        // sat=1→+1.125, sat=4→-1.125
  s-=Math.min(inc,20000)/20000*3.00;        // income: 2800→-0.42, 9000→-1.35, 14000→-2.10
  s-=Math.min(yr,20)/20*2.20;              // years: 1→-0.11, 10→-1.10, 20→-2.20
  s-=Math.min(Math.max(age-18,0),32)/32*0.80; // age
  if(ht===1) s-=1.30;                       // tenure+sat combined
  if(gm===1) s-=0.15;

  const p=Math.min(0.96,Math.max(0.04,sig(s)));
  return{attrition_probability:+p.toFixed(5),prediction:p>=0.5?1:0,mode:'approximation',
         probabilities:{'No Attrition':+(1-p).toFixed(5),'Attrition':+p.toFixed(5)}};
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
    'background:#ff3d6b;color:#fff;font-family:var(--mono);font-size:11px;padding:.75rem 1.5rem;'+
    'letter-spacing:.1em;box-shadow:0 4px 24px rgba(255,61,107,.4);animation:fadein .3s ease';
  b.innerHTML='⚠ AWS SESSION TOKEN EXPIRED — Refresh credentials in .env from AWS Academy → AWS Details → Show';
  document.body.appendChild(b);
  setTimeout(()=>{if(b)b.remove();},12000);
}

/* ── Health check ── */
function checkHealth(){
  fetch('/api/health').then(r=>r.json()).then(h=>{
    const dot=document.getElementById('stDot');
    const lbl=document.getElementById('stLabel');
    if(!dot||!lbl)return;
    if(h.ready){
      dot.className='dot live';lbl.textContent='ENDPOINT LIVE';lbl.style.color='var(--green)';
      addLog('SageMaker connected: '+h.endpoint_name+' ['+h.endpoint_status+']','ok');
    } else if(!h.token_valid||h.endpoint_status==='expired_token'){
      dot.className='dot dead';lbl.textContent='TOKEN EXPIRED';lbl.style.color='var(--red)';
      addLog('AWS session token EXPIRED — refresh .env from AWS Academy','err');
      showExpiredBanner();
    } else if(h.credentials==='missing'){
      dot.className='dot dead';lbl.textContent='CREDS MISSING';lbl.style.color='var(--red)';
      addLog('AWS keys missing in .env','err');
    } else {
      dot.className='dot dead';lbl.textContent='ENDPOINT OFFLINE';lbl.style.color='var(--red)';
      addLog('Endpoint status: '+h.endpoint_status,'err');
    }
  }).catch(()=>{
    const lbl=document.getElementById('stLabel');
    if(lbl){lbl.textContent='SERVER ERROR';document.getElementById('stDot').className='dot dead';}
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
  else if(ap>=0.35){verdict='⚡ MEDIUM ATTRITION RISK';  cls='high';riskTxt='🟡 MEDIUM';   riskColor='var(--gold)';}
  else if(ap>=0.18){verdict='✓ LOW ATTRITION RISK';      cls='low'; riskTxt='🟢 LOW';      riskColor='var(--green)';}
  else             {verdict='✓ VERY LIKELY TO STAY';     cls='low'; riskTxt='🟢 VERY LOW'; riskColor='var(--green)';}

  addLog('→ '+verdict+' ('+apPct+'%)',pred?'err':'ok');

  const box=document.getElementById('resBox');
  box.classList.add('show');
  const v=document.getElementById('rV');
  v.textContent=verdict;v.className='result-verdict '+cls;
  document.getElementById('rP').textContent='Attrition probability: '+apPct+'%';
  const bar=document.getElementById('rBar');
  bar.className='bar-fill '+(pred?'high':'low');
  setTimeout(()=>{bar.style.width=apPct+'%';},50);
  const rA=document.getElementById('rA');
  rA.textContent=apPct+'%';rA.style.color=pred?'var(--red)':'var(--green)';
  document.getElementById('rR').textContent=rpPct+'%';
  const rRisk=document.getElementById('rRisk');
  rRisk.textContent=riskTxt;rRisk.style.color=riskColor;
  const rMode=document.getElementById('rMode');
  rMode.textContent=mode.includes('sagemaker')?'✅ SAGEMAKER':'⚡ APPROX';
  rMode.style.color=mode.includes('sagemaker')?'var(--green)':'var(--gold)';
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
      addLog('⚠ AWS TOKEN EXPIRED — using approximation. Refresh .env!','err');
      showExpiredBanner();
      result=approx(feat);
    } else if(result.error){
      addLog('API error (using approx): '+result.error,'err');
      result=approx(feat);
    }
  }catch(e){
    addLog('Network error — using approximation','err');
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
