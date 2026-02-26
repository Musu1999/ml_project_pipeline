/* frontend/static/js/batch.js — Batch predict page logic */

const BATCH_EXAMPLE = {
  "employees": [
    {"name": "Alice Johnson",  "features": [0,1,28,3200,1,1,2,2]},
    {"name": "Bob Smith",      "features": [1,0,45,8500,0,1,15,4]},
    {"name": "Carol Williams", "features": [0,0,35,5500,0,0,7,3]},
    {"name": "David Brown",    "features": [0,1,32,4200,1,0,4,1]},
    {"name": "Emma Davis",     "features": [1,0,52,12000,0,1,20,4]}
  ]
};

function addLog(msg,cls=''){
  const t=document.getElementById('batchTerm');
  if(!t)return;
  const d=document.createElement('div');
  d.className='log-line '+cls;
  d.textContent=`[${new Date().toLocaleTimeString('en-GB',{hour12:false})}] ${msg}`;
  t.appendChild(d);t.scrollTop=t.scrollHeight;
}

function loadExample(){
  const ta=document.getElementById('batchInput');
  if(ta) ta.value=JSON.stringify(BATCH_EXAMPLE,null,2);
  addLog('Example data loaded — edit names/features then click Run Batch','inf');
}

function getRiskClass(risk){
  if(risk==='HIGH')   return 'risk-high';
  if(risk==='MEDIUM') return 'risk-medium';
  return 'risk-low';
}
function getRiskEmoji(risk){
  if(risk==='HIGH')   return '🔴';
  if(risk==='MEDIUM') return '🟡';
  return '🟢';
}

async function runBatch(){
  const ta=document.getElementById('batchInput');
  const btn=document.getElementById('batchBtn');
  const btxt=document.getElementById('batchBtnTxt');

  let payload;
  try{
    payload=JSON.parse(ta.value);
  }catch(e){
    addLog('Invalid JSON: '+e.message,'err');
    return;
  }
  if(!payload.employees||!Array.isArray(payload.employees)){
    addLog('JSON must have "employees" array','err');return;
  }

  btn.disabled=true;btxt.textContent='⟳ PROCESSING '+payload.employees.length+' EMPLOYEES...';
  addLog('Batch predict: '+payload.employees.length+' employees','inf');

  let result;
  try{
    const r=await fetch('/api/predict/batch',{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    result=await r.json();
    if(result.error){addLog('API error: '+result.error,'err');btn.disabled=false;btxt.textContent='▶ RUN BATCH';return;}
  }catch(e){
    addLog('Network error: '+e.message,'err');btn.disabled=false;btxt.textContent='▶ RUN BATCH';return;
  }

  btn.disabled=false;btxt.textContent='▶ RUN BATCH';
  addLog(`Done: ${result.success} success, ${result.failed} failed`,'ok');

  /* Summary */
  const sum=result.summary||{};
  const el=id=>document.getElementById(id);
  if(el('bTotal'))  el('bTotal').textContent  = result.total   ||0;
  if(el('bHigh'))   {el('bHigh').textContent  = sum.high_risk  ||0; el('bHigh').style.color='var(--red)';}
  if(el('bMed'))    {el('bMed').textContent   = sum.medium_risk||0; el('bMed').style.color='var(--gold)';}
  if(el('bLow'))    {el('bLow').textContent   = sum.low_risk   ||0; el('bLow').style.color='var(--green)';}

  /* Table */
  const tbody=document.getElementById('batchTbody');
  if(!tbody)return;
  tbody.innerHTML='';
  (result.predictions||[]).forEach(p=>{
    const ap=(p.attrition_probability*100).toFixed(1);
    const rp=((1-p.attrition_probability)*100).toFixed(1);
    const rc=getRiskClass(p.risk_level);
    const re=getRiskEmoji(p.risk_level);
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td>${p.name}</td>
      <td class="${p.prediction===1?'risk-high':'risk-low'}">${p.predicted_class}</td>
      <td>${ap}%</td>
      <td>${rp}%</td>
      <td class="${rc}">${re} ${p.risk_level}</td>`;
    tbody.appendChild(tr);
  });
  (result.errors||[]).forEach(e=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${e.name}</td><td colspan="4" class="risk-high">ERROR: ${e.error}</td>`;
    tbody.appendChild(tr);
  });

  document.getElementById('batchResults').style.display='block';
}

/* ── Health check ── */
function checkHealth(){
  fetch('/api/health').then(r=>r.json()).then(h=>{
    const dot=document.getElementById('stDot');
    const lbl=document.getElementById('stLabel');
    if(!dot||!lbl)return;
    if(h.ready){
      dot.className='dot live';lbl.textContent='ENDPOINT LIVE';lbl.style.color='var(--green)';
    }else{
      dot.className='dot dead';
      lbl.textContent=h.credentials==='missing'?'CREDS MISSING':'ENDPOINT OFFLINE';
      lbl.style.color='var(--red)';
    }
  }).catch(()=>{});
}

window.addEventListener('load',()=>{
  checkHealth();
  addLog('Batch prediction page ready','ok');
  addLog('Click "Load Example" to see input format','inf');
});
