/* frontend/static/js/app.js — Single predict page logic */

const SAMPLES = [
  {name:"Young Overtime · Single",   feat:[0,1,28,3200,1,1,2,2]},
  {name:"Senior Satisfied Manager",  feat:[1,0,45,8500,0,1,15,4]},
  {name:"Mid-Career Professional",   feat:[0,0,35,5500,0,0,7,3]},
  {name:"Unhappy Overtime Staff",    feat:[0,1,32,4200,1,0,4,1]},
  {name:"Tenured Expert (20yr)",     feat:[1,0,52,12000,0,1,20,4]},
];
const FEATS = [
  {n:"OverTime",v:.312},{n:"Income",v:.218},{n:"Age",v:.142},{n:"YearsAtCo",v:.118},
  {n:"JobSat",v:.098},{n:"Marital",v:.062},{n:"Tenure+Sat",v:.031},{n:"Gender",v:.019},
];
const BARS = [
  {l:"Sat 1",r:88,c:"#ff3d6b"},{l:"Sat 2",r:62,c:"#ff8c42"},
  {l:"Sat 3",r:38,c:"#ffd166"},{l:"Sat 4",r:18,c:"#a8ff3e"},
];

/* ── Sigmoid fallback ── */
function sig(x){return 1/(1+Math.exp(-x))}
function approx(f){
  const[ht,ot,age,inc,ms,gm,yr,sat]=f;
  let s=-1.8;
  s+=ot*1.65;s+=ms*.72;s-=inc/3000*.55;s-=sat*.48;s-=ht*.60;s-=yr/5*.30;s-=(age-30)/10*.25;
  const p=Math.min(.97,Math.max(.03,sig(s)));
  return{attrition_probability:p,prediction:p>=.5?1:0,mode:'approximation',
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

/* ── Health check ── */
function checkHealth(){
  fetch('/api/health').then(r=>r.json()).then(h=>{
    const dot=document.getElementById('stDot');
    const lbl=document.getElementById('stLabel');
    if(!dot||!lbl)return;
    if(h.ready){
      dot.className='dot live';lbl.textContent='ENDPOINT LIVE';lbl.style.color='var(--green)';
      addLog('SageMaker connected: '+h.endpoint_name+' ['+h.endpoint_status+']','ok');
    } else {
      dot.className='dot dead';
      lbl.textContent=h.credentials==='missing'?'CREDS MISSING':'ENDPOINT OFFLINE';
      lbl.style.color='var(--red)';
      addLog('WARNING: '+(h.credentials==='missing'?'AWS keys missing in .env':'endpoint status: '+h.endpoint_status),'err');
    }
  }).catch(()=>{
    const lbl=document.getElementById('stLabel');
    if(lbl){lbl.textContent='SERVER ERROR';document.getElementById('stDot').className='dot dead';}
  });
}

/* ── Predict ── */
async function runPredict(){
  const feat=[
    +document.getElementById('fHt').value,
    +document.getElementById('fOt').value,
    +document.getElementById('fAge').value,
    +document.getElementById('fInc').value,
    +document.getElementById('fMs').value,
    +document.getElementById('fGm').value,
    +document.getElementById('fYrs').value,
    +document.getElementById('fSat').value,
  ];
  const btn=document.getElementById('pBtn');
  const btxt=document.getElementById('bTxt');
  btn.disabled=true;btxt.textContent='⟳ COMPUTING...';
  addLog('Inference — features: ['+feat.join(',')+']','inf');

  let result;
  try{
    const r=await fetch('/api/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({features:feat})});
    result=await r.json();
    if(result.error){addLog('API error: '+result.error,'err');result=approx(feat);}
  }catch(e){
    addLog('Network error — using approximation','err');
    result=approx(feat);
  }

  btn.disabled=false;btxt.textContent='▶ RUN PREDICTION';

  const isHigh=result.prediction===1||result.predicted_class==='Attrition';
  const ap=+(result.attrition_probability||0);
  const rp=+(result.probabilities?.['No Attrition']||(1-ap));
  const apPct=(ap*100).toFixed(1),rpPct=(rp*100).toFixed(1);
  const mode=result.mode||'sagemaker_live';

  addLog('Result: '+(isHigh?'ATTRITION':'RETAINED')+' ('+apPct+'%)',isHigh?'err':'ok');

  const box=document.getElementById('resBox');
  box.classList.add('show');
  const v=document.getElementById('rV');
  v.textContent=isHigh?'⚠ HIGH ATTRITION RISK':'✓ LIKELY TO STAY';
  v.className='result-verdict '+(isHigh?'high':'low');
  document.getElementById('rP').textContent='Attrition probability: '+apPct+'%';
  const bar=document.getElementById('rBar');
  bar.className='bar-fill '+(isHigh?'high':'low');
  setTimeout(()=>{bar.style.width=apPct+'%'},50);
  document.getElementById('rA').textContent=apPct+'%';
  document.getElementById('rA').style.color=isHigh?'var(--red)':'var(--green)';
  document.getElementById('rR').textContent=rpPct+'%';
  document.getElementById('rRisk').textContent=isHigh?'🔴 HIGH':'🟢 LOW';
  document.getElementById('rRisk').style.color=isHigh?'var(--red)':'var(--green)';
  document.getElementById('rMode').textContent=mode.includes('sagemaker')?'SAGEMAKER':'APPROX';
  document.getElementById('rMode').style.color=mode.includes('sagemaker')?'var(--green)':'var(--gold)';
}

/* ── Render Profiles ── */
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
      document.getElementById('fHt').value=ht;
      document.getElementById('fOt').value=ot;
      document.getElementById('fAge').value=age;
      document.getElementById('fInc').value=inc;
      document.getElementById('fMs').value=ms;
      document.getElementById('fGm').value=gm;
      document.getElementById('fYrs').value=yr;
      document.getElementById('fSat').value=sat;
      runPredict();
    };
    list.appendChild(d);
  });
}

/* ── Render Feature Importance ── */
function renderFeats(){
  const list=document.getElementById('featList');
  if(!list)return;
  const mx=FEATS[0].v;
  FEATS.forEach(f=>{
    const pct=(f.v/mx*100).toFixed(0);
    const d=document.createElement('div');
    d.className='feat-row';
    d.innerHTML=`<div class="feat-name">${f.n}</div>
      <div class="feat-bar"><div class="feat-bar-fill" style="width:0%" data-w="${pct}%"></div></div>
      <div class="feat-val">${(f.v*100).toFixed(1)}%</div>`;
    list.appendChild(d);
  });
  setTimeout(()=>document.querySelectorAll('.feat-bar-fill').forEach(e=>e.style.width=e.dataset.w),400);
}

/* ── Render Bar Chart ── */
function renderBars(){
  const c=document.getElementById('barChart');
  if(!c)return;
  const mx=Math.max(...BARS.map(b=>b.r));
  BARS.forEach(b=>{
    const h=(b.r/mx*86).toFixed(0);
    const d=document.createElement('div');
    d.className='bar-col';
    d.innerHTML=`<div class="bar-col-bar" style="height:${h}px;background:${b.c};box-shadow:0 0 8px ${b.c}55" title="${b.r}%"></div>
      <div class="bar-col-label">${b.l}<br>${b.r}%</div>`;
    c.appendChild(d);
  });
}

/* ── Donut Chart ── */
function animateDonut(){
  const C=2*Math.PI*38,al=C*.161,rl=C*.839;
  const da=document.getElementById('dAttr');
  const dr=document.getElementById('dRet');
  if(!da||!dr)return;
  da.setAttribute('stroke-dasharray',`${al} ${C}`);
  dr.setAttribute('stroke-dasharray',`${rl} ${C}`);
  dr.setAttribute('stroke-dashoffset',`-${al}`);
}

/* ── Init ── */
window.addEventListener('load',()=>{
  setTimeout(()=>{const l=document.getElementById('loader');if(l)l.classList.add('gone')},1000);
  renderProfiles();
  renderFeats();
  renderBars();
  setTimeout(animateDonut,600);
  checkHealth();
  addLog('AttritionIQ dashboard ready','ok');
});
