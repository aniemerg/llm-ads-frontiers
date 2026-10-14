// Live inference has one active input snapshot. Inspector previews never mutate it.
const $ = selector => document.querySelector(selector);
const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n,d=4) => Number(n).toFixed(d);
const probability = p => p===0?'0%':p<.00001?`${(p*100).toExponential(2)}%`:`${fmt(p*100,3)}%`;
const busyStates = ['loading','prefilling','generating','validating','no-ad-generating','no-ad-validating'];
const state = {inputs:null,runId:null,status:'connecting',events:[],manifest:null,result:null,error:null,
  noAd:null,selected:-1,follow:true,cfA:1,cfB:1,page:0,tokenQuery:'',inspect:null,dirty:false};
let mounted=false,pollTimer=null,inspectTimer=null,inspectEpoch=0,submitting=false;
let inspectionKey='',polling=false;

async function api(url,options){
  const response=await fetch(url,options);
  const value=await response.json();
  if(!response.ok){const error=new Error(value.error||`Request failed (${response.status})`);error.status=response.status;throw error;}
  return value;
}
const isBusy=()=>submitting||busyStates.includes(state.status);
const tokenText=e=>e.token_text==='<|im_end|>'?'[EOS]':e.token_text||`#${e.token_id}`;
const fileLink=(name,label)=>`<a href="/api/live/file?run_id=${encodeURIComponent(state.runId)}&name=${encodeURIComponent(name)}" target="_blank" rel="noopener" class="source-link">${label}</a>`;

function editor(ad,i){
  const side=i?'B':'A';
  return `<section class="panel live-ad ${i?'b':'a'}"><div class="live-ad-heading"><span class="eyebrow">ADVERTISER ${side}</span><label class="live-bid">Bid / token <input data-ad="${i}" data-field="bid" aria-label="Advertiser ${side} bid" type="number" min="0" max="1000" step="any" required value="${ad.bid}"></label></div><label>Brand name<input data-ad="${i}" data-field="brand" aria-label="Advertiser ${side} brand name" maxlength="160" required value="${esc(ad.brand)}"></label><label>Allowed facts<textarea data-ad="${i}" data-field="facts" aria-label="Advertiser ${side} allowed facts" maxlength="6000" rows="5">${esc(ad.facts)}</textarea></label><label>Your advertising objective<textarea data-ad="${i}" data-field="objective" aria-label="Advertiser ${side} advertising objective" maxlength="6000" rows="4">${esc(ad.objective)}</textarea></label><p class="small muted">Both contexts see these facts. Only ${side}’s context sees this advertising objective. Bids stay out of prompts.</p></section>`;
}

function form(){
  const v=state.inputs;
  return `<fieldset id="live-fields"><legend class="sr-only">Live generation inputs</legend><div class="live-editors">${editor(v.advertisers[0],0)}<section class="panel live-shared"><p class="eyebrow">ONE SHARED TASK</p><h2>What should we write?</h2><label>User prompt<textarea id="live-prompt" maxlength="6000" rows="7" required>${esc(v.prompt)}</textarea></label><p class="small muted">Edit the prompt, briefs, or bids. Press Refresh inference when you’re ready. One new model run replaces the active result.</p><details><summary>Shared instructions for ad contexts</summary><div><label>Instructions for both contexts<textarea id="live-instructions" maxlength="8000" rows="9">${esc(v.instructions)}</textarea></label><p class="small muted">These rules apply only to advertiser contexts. The no-ad reference receives only your user prompt. Exact rendered prompts are available below.</p></div></details><div class="live-settings"><label>Seed<input id="live-seed" type="number" min="0" max="4294967295" step="1" required value="${v.seed}"></label><label>Temperature<input id="live-temperature" type="number" min="0.05" max="2" step="any" required value="${v.temperature}"></label><label>Token cap<input id="live-cap" type="number" min="1" max="96" step="1" required value="${v.max_tokens}"></label></div><p class="small muted">Full-vocabulary sampling · no top-k/top-p filter. Both bids zero selects the separate no-ad path.</p></section>${editor(v.advertisers[1],1)}</div></fieldset>`;
}

export async function mountLive(){
  mounted=true;
  $('#content').innerHTML=`<div class="live-heading"><div><p class="eyebrow">LIVE LAB / LOCAL MODEL INFERENCE</p><h1>One message. <em>Your inputs.</em></h1><p>Qwen3-4B on this Mac. Refresh to generate the ad response and a prompt-only reference, then inspect how the probabilities differ.</p></div><span class="badge live-badge">LIVE MODEL</span></div><div class="live-toolbar"><div id="live-status" role="status" aria-live="polite"><span class="spinner"></span> Connecting to the local model service…</div><div class="controls"><button id="live-new-seed">Next seed</button><button id="live-rerun" class="primary">Refresh inference ↻</button></div></div><div id="live-form"></div><div id="live-notice" role="alert"></div><div id="live-output"></div><div id="live-inspector"></div><div id="live-provenance"></div>`;
  $('#live-rerun').addEventListener('click',startRun);
  $('#live-new-seed').addEventListener('click',()=>{
    if(isBusy()||!state.inputs)return;
    state.inputs.seed=(state.inputs.seed+1)%4294967296;$('#live-seed').value=state.inputs.seed;invalidate();
  });
  if(state.inputs){renderForm();renderStatus();renderOutput();}
  try{
    if(state.dirty||isBusy()&&state.runId){
      if(!polling&&!pollTimer&&!state.dirty)poll();
      return;
    }
    const snapshot=await api('/api/live');
    if(snapshot.status==='idle'){
      state.inputs=snapshot.defaults;state.status='idle';renderForm();renderStatus();renderOutput();
    }else{applySnapshot(snapshot,true);renderForm();renderStatus();renderOutput();schedulePoll();}
  }catch(error){state.error=error.message;state.status='connection-error';renderStatus();}
}

export function unmountLive(){mounted=false;inspectionKey='';inspectEpoch++;}

function renderForm(){
  if(!mounted||!state.inputs)return;
  $('#live-form').innerHTML=form();
  $('#live-fields').addEventListener('input',()=>{
    if(isBusy())return;
    state.inputs={
      prompt:$('#live-prompt').value,instructions:$('#live-instructions').value,
      advertisers:[0,1].map(i=>Object.fromEntries(['brand','facts','objective','bid'].map(key=>{
        const el=document.querySelector(`[data-ad="${i}"][data-field="${key}"]`);
        return [key,key==='bid'?(el.value===''?null:Number(el.value)):el.value];
      }))),
      seed:$('#live-seed').value===''?null:Number($('#live-seed').value),
      temperature:$('#live-temperature').value===''?null:Number($('#live-temperature').value),
      max_tokens:$('#live-cap').value===''?null:Number($('#live-cap').value),
    };
    invalidate();
  });
  renderStatus();
}

function invalidate(){
  clearTimeout(pollTimer);pollTimer=null;
  state.dirty=true;state.status='stale';state.events=[];state.result=null;state.manifest=null;state.noAd=null;
  state.error=null;state.selected=-1;state.inspect=null;state.runId=null;inspectionKey='';inspectEpoch++;
  renderStatus();renderOutput();
}

function clientError(){
  const v=state.inputs;
  if(!v||!v.prompt.trim())return 'Enter a shared prompt to start inference.';
  if(v.advertisers.some(ad=>!ad.brand.trim()))return 'Give both advertisers a brand name.';
  if(v.advertisers.some(ad=>ad.bid===null||!Number.isFinite(ad.bid)||ad.bid<0||ad.bid>1000))return 'Bids must be between 0 and 1000 simulated credits.';
  if(!Number.isInteger(v.seed)||v.seed<0||v.seed>4294967295)return 'Enter an integer seed between 0 and 4294967295.';
  if(v.temperature===null||!Number.isFinite(v.temperature)||v.temperature<.05||v.temperature>2)return 'Temperature must be between 0.05 and 2.';
  if(!Number.isInteger(v.max_tokens)||v.max_tokens<1||v.max_tokens>96)return 'The token cap must be an integer between 1 and 96.';
  return null;
}

async function startRun(){
  if(isBusy())return;
  const error=clientError();
  if(error){state.error=error;state.status='invalid';renderStatus();return;}
  submitting=true;state.dirty=false;state.status='loading';state.error=null;state.events=[];
  state.selected=-1;state.follow=true;state.result=null;state.manifest=null;state.runId=null;state.noAd=null;
  inspectionKey='';inspectEpoch++;renderStatus();renderOutput();
  try{
    const response=await api('/api/live/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state.inputs)});
    state.runId=response.run_id;submitting=false;await poll();
  }catch(error){
    submitting=false;
    if(error.status===409){
      const active=await api('/api/live');applySnapshot(active,true);renderForm();renderStatus();renderOutput();schedulePoll();
    }else{state.error=error.message;state.status='error';renderStatus();renderOutput();}
  }
}

function applySnapshot(snapshot,replace=false){
  const changed=state.runId!==snapshot.run_id;
  if(replace||changed){state.events=[];state.inputs=snapshot.inputs;state.selected=-1;state.follow=true;inspectionKey='';inspectEpoch++;}
  state.runId=snapshot.run_id;state.status=snapshot.status;state.error=snapshot.error;
  state.result=snapshot.result;state.manifest=snapshot.manifest;state.noAd=snapshot.no_ad||null;
  for(const event of snapshot.events||[]){if(event.step===state.events.length)state.events.push(event);}
  if(state.follow)state.selected=state.events.length-1;
}

function schedulePoll(){
  clearTimeout(pollTimer);pollTimer=null;
  if(busyStates.includes(state.status))pollTimer=setTimeout(poll,250);
}

async function poll(){
  if(polling||!state.runId)return;
  polling=true;
  const id=state.runId;
  try{
    const snapshot=await api(`/api/live?run_id=${encodeURIComponent(id)}&after=${state.events.length}`);
    if(id!==state.runId||state.dirty)return;
    applySnapshot(snapshot);renderStatus();renderOutput();
  }catch(error){
    if(error.status===409){const snapshot=await api('/api/live');applySnapshot(snapshot,true);renderForm();renderStatus();renderOutput();}
    else{state.error=`Connection interrupted: ${error.message}. Checking the active run again…`;renderStatus();}
  }finally{polling=false;schedulePoll();}
}

function renderStatus(){
  if(!mounted)return;
  const busy=isBusy();
  if($('#live-fields'))$('#live-fields').disabled=busy||state.status==='connecting';
  $('#live-rerun').disabled=busy||!state.inputs;
  $('#live-new-seed').disabled=busy||!state.inputs;
  const captions={connecting:'Connecting to the local service…',idle:'Ready for real inference',
    stale:'Inputs changed · press Refresh inference to run',
    loading:'Loading the cached model · inputs locked',prefilling:'Reading both prompts · inputs locked',
    generating:`Generating token ${state.events.length} / ${state.inputs?state.inputs.max_tokens:96} · inputs locked`,
    validating:'Checking advertiser and reference caches · inputs locked',
    'no-ad-generating':`Generating the separate no-ad answer · ${(state.noAd&&state.noAd.events||[]).length} tokens · inputs locked`,
    'no-ad-validating':'Checking the no-ad answer cache · inputs locked',complete:'Complete · refresh manually when you’re ready',
    error:'Run failed · edit an input, then press Refresh inference',invalid:'Waiting for valid inputs',
    'connection-error':'Local service unavailable · reload to reconnect'};
  $('#live-status').innerHTML=`${busy||state.status==='connecting'?'<span class="spinner"></span>':'<span class="status-dot"></span>'}<span>${esc(captions[state.status]||state.status)}</span>`;
  $('#live-notice').innerHTML=state.error?`<div class="callout warning"><strong>${state.status==='error'?'This run did not validate.':'Please check this input.'}</strong><pre class="live-error">${esc(state.error)}</pre>${state.events.length?'<p>The generated tokens and charges below are preserved as a failed live attempt.</p>':''}</div>`:'';
}

function renderOutput(){
  if(!mounted)return;
  if(!state.events.length){
    $('#live-output').innerHTML=`<section class="panel live-empty"><span class="${isBusy()?'spinner':'empty-glyph'}">${isBusy()?'':'↳'}</span><h2>${state.dirty?'Inputs changed. Ready when you are.':isBusy()?'The model is working':'Your active response appears here'}</h2><p>${state.dirty?'Press Refresh inference to generate from the edited inputs. The previous result was cleared because it is stale.':isBusy()?'Tokens, probabilities, and simulated charges will appear as they are generated.':'Edit the inputs above, then press Refresh inference to run the model.'}</p></section>`;
    $('#live-inspector').innerHTML='';$('#live-provenance').innerHTML='';return;
  }
  const latest=state.events[state.events.length-1],r=state.result;
  const baseline=latest.branch==='baseline';
  const status=state.status==='error'?'FAILED LIVE ATTEMPT':state.result?'LIVE RESULT · CACHE CHECK PASSED':'LIVE GENERATION · VALIDATION PENDING';
  const outputExists=Boolean($('#live-message'));
  if(!outputExists){$('#live-output').innerHTML=`<section class="panel live-response"><div class="replay-head"><h3 id="live-output-title"></h3><span class="badge live-badge" id="live-output-badge"></span></div><p id="live-message" class="output-text"></p><div class="live-metrics" id="live-metrics"></div><div class="section-title"><h3>Inspect any generated token</h3><button id="live-follow">Follow newest token</button></div><div class="token-ribbon" id="live-tokens"></div><div class="legend"><span><i class="dot shared"></i>Shared · free</span><span><i class="dot a"></i>A branch</span><span><i class="dot b"></i>B branch</span><span>Branch labels describe sampling, not authorship.</span></div><p class="small muted">Original model output, with no text repair. Token labels may show partial words or byte fragments; the message is decoded from the complete ID sequence.</p></section>`;
    $('#live-follow').addEventListener('click',()=>{state.follow=true;state.selected=state.events.length-1;inspectionKey='';renderOutput();});}
  $('#live-output-title').textContent=baseline?'No-ad response · both bids are zero':'Ad-influenced response';
  $('#live-output-badge').textContent=status;
  $('#live-message').textContent=latest.decoded_text_so_far||'[No visible text yet]';
  $('#live-metrics').innerHTML=[['Tokens',state.events.length],['Generation',`${fmt(r?r.seconds:latest.elapsed_seconds,2)} s`],['A total charge',fmt(latest.cumulative_a,6)],['B total charge',fmt(latest.cumulative_b,6)],['Stop',r?(r.stop_reason==='length'?'token cap · truncated':r.stop_reason):state.status==='error'?'failed':'in progress']].map(([k,v])=>`<div><span>${k}</span><strong>${esc(v)}</strong></div>`).join('');
  $('#live-follow').textContent=state.follow?'Following newest ✓':'Follow newest token';
  $('#live-follow').setAttribute('aria-pressed',String(state.follow));
  $('#live-tokens').innerHTML=state.events.map((e,i)=>`<button class="token branch-${e.branch}" data-live-token="${i}" aria-pressed="${i===state.selected}" title="Token ${i+1} · ID ${e.token_id} · ${e.branch}">${esc(tokenText(e))}</button>`).join('');
  document.querySelectorAll('[data-live-token]').forEach(button=>button.addEventListener('click',()=>{
    state.follow=false;state.selected=Number(button.dataset.liveToken);inspectionKey='';renderOutput();
  }));
  renderInspector();renderProvenance();
}

function renderInspector(){
  if(!mounted||state.selected<0)return;
  const e=state.events[state.selected],key=`${state.runId}:${state.selected}`;
  if(inspectionKey===key)return;
  inspectionKey=key;state.cfA=e.bid_a===undefined?0:e.bid_a;state.cfB=e.bid_b===undefined?0:e.bid_b;
  state.page=0;state.tokenQuery='';state.inspect=null;
  const baseline=e.branch==='baseline',shared=e.branch==='shared';
  $('#live-inspector').innerHTML=`<div class="live-inspector-heading"><div><p class="eyebrow">TOKEN INSPECTOR / ACTUAL FULL DISTRIBUTIONS</p><h2>Token ${state.selected+1}: <span class="mono">${esc(tokenText(e))}</span></h2></div><span class="badge">ID ${e.token_id}</span></div><div class="inspector-grid live-inspector-grid"><section class="panel"><div class="replay-head"><h3>Probability explorer</h3><span id="live-prob-mode" class="badge">ACTUAL RUN</span></div><p class="small muted section-gap">${baseline?'One no-ad context, with no auction or advertiser payments.':'p_A and p_B come from the advertiser contexts. Ad mixture q = (a × p_A + b × p_B) / (a + b) is the distribution sampled by the auction. It is not the no-ad baseline.'}</p><div id="live-no-ad-shift"></div><div id="live-pinned"></div><div class="prob-tools"><label>Look up token ID<input id="live-token-query" type="number" min="0" step="1" placeholder="e.g. 1910" aria-label="Probability lookup token ID"></label><button id="live-lookup">Look up</button><button id="live-clear-query">Top probabilities</button></div><div id="live-prob-table"><p class="small muted">Loading full probabilities…</p></div><div class="prob-pagination"><button id="live-page-prev">← Previous 20</button><span id="live-page-label" class="small muted"></span><button id="live-page-next">Next 20 →</button></div><p class="small muted section-gap">The table is paginated for display. Sampling uses the full vocabulary with no filtering or renormalization of displayed rows.</p></section>
  <section class="panel"><h3>The actual decision</h3><p class="small muted">Prefix before this token:</p><div class="live-prefix">${esc(state.selected?state.events[state.selected-1].decoded_text_so_far:'[Start of the assistant response]')}</div>${baseline?'<div class="callout">Both bids are zero. This is a separate single-context, no-ad generation. Counterfactual auctions are unavailable on this path.</div>':`<div class="mass-bar"><span class="common" style="width:${e.shared_mass*100}%">${e.shared_mass>.15?fmt(e.shared_mass*100,0)+'%':''}</span><span class="contested" style="width:${e.disagreement_mass*100}%">${e.disagreement_mass>.15?fmt(e.disagreement_mass*100,0)+'%':''}</span></div><p class="small muted">Shared ${probability(e.shared_mass)} · disagreement ${probability(e.disagreement_mass)}</p><div class="kv"><span>Branch</span><strong>${shared?'Shared · free':`${e.branch} wins contest`}</strong></div><div class="kv"><span>Branch draw / common mass</span><strong>${fmt(e.tape.branch,6)} / ${fmt(e.shared_mass,6)}</strong></div>${shared?'':`<div class="kv"><span>A residual candidate</span><strong>${esc(e.candidate_a_text)} · #${e.candidate_a}</strong></div><div class="kv"><span>B residual candidate</span><strong>${esc(e.candidate_b_text)} · #${e.candidate_b}</strong></div><div class="kv"><span>Contest draw u / A weight</span><strong>${fmt(e.tape.contest,6)} / ${fmt(e.bid_a/(e.bid_a+e.bid_b),6)}</strong></div>`}<div class="live-step-charges"><div><span>A paid this step</span><strong>${fmt(e.payment_a,6)}</strong></div><div><span>B paid this step</span><strong>${fmt(e.payment_b,6)}</strong></div></div><p class="small muted">${shared?'A shared draw is independent of bids, so both charges are zero.':`${e.branch} pays the critical bid: ${e.branch==='A'?'b × u / (1 − u)':'a × (1 − u) / u'}. This payment is bounded by its fixed per-token bid.`}</p><div class="frozen"><p class="eyebrow">FROZEN-PREFIX MATH / NOT A NEW GENERATION</p><h3>Try another bid for this token only</h3><p class="muted">This uses the same full distributions and random tape in the Python reference auction. Refresh inference applies the main inputs above to a new model run; these preview controls leave the active message and charges unchanged.</p><div class="controls"><label class="live-cf-label">Preview A bid<input id="live-cf-a" type="number" min="0" max="1000" step="any" value="${e.bid_a}"></label><label class="live-cf-label">Preview B bid<input id="live-cf-b" type="number" min="0" max="1000" step="any" value="${e.bid_b}"></label><button id="live-cf-reset">Reset</button></div><div id="live-cf-outcome" aria-live="polite"></div></div><details><summary>Exact random tape</summary><div><pre>${esc(JSON.stringify(e.tape,null,2))}</pre></div></details>`}</section></div>`;
  if(!baseline){
    ['a','b'].forEach(k=>$(`#live-cf-${k}`).addEventListener('input',()=>{
      state[k==='a'?'cfA':'cfB']=$(`#live-cf-${k}`).value===''?null:Number($(`#live-cf-${k}`).value);state.page=0;
      // Invalidate immediately, not when the debounced request starts.
      inspectEpoch++;clearTimeout(inspectTimer);showInspectPending();inspectTimer=setTimeout(fetchInspection,250);
    }));
    $('#live-cf-reset').addEventListener('click',()=>{state.cfA=e.bid_a;state.cfB=e.bid_b;$('#live-cf-a').value=e.bid_a;$('#live-cf-b').value=e.bid_b;state.page=0;fetchInspection();});
  }
  $('#live-lookup').addEventListener('click',()=>{state.tokenQuery=$('#live-token-query').value;state.page=0;fetchInspection();});
  $('#live-token-query').addEventListener('keydown',e=>{if(e.key==='Enter')$('#live-lookup').click();});
  $('#live-clear-query').addEventListener('click',()=>{state.tokenQuery='';$('#live-token-query').value='';state.page=0;fetchInspection();});
  $('#live-page-prev').addEventListener('click',()=>{state.page=Math.max(0,state.page-1);fetchInspection();});
  $('#live-page-next').addEventListener('click',()=>{state.page++;fetchInspection();});
  fetchInspection();
}

function showInspectPending(){
  if(!mounted)return;
  if($('#live-cf-outcome'))$('#live-cf-outcome').innerHTML='<p class="small muted">Calculating the frozen counterfactual…</p>';
  $('#live-prob-table').innerHTML='<p class="small muted">Updating probabilities…</p>';
  $('#live-pinned').innerHTML='';
}

async function fetchInspection(){
  const epoch=++inspectEpoch;
  if(!mounted||!state.runId||state.selected<0)return;
  const e=state.events[state.selected],baseline=e.branch==='baseline';
  if(!baseline&&(state.cfA===null||state.cfB===null||state.cfA<0||state.cfB<0||!Number.isFinite(state.cfA+state.cfB)||state.cfA+state.cfB===0)){
    if($('#live-cf-outcome'))$('#live-cf-outcome').innerHTML='<p class="small">Use nonnegative preview bids with at least one positive bid. The separate no-ad path requires a new live run.</p>';
    $('#live-prob-table').innerHTML='<p class="small muted">No auction exists for this preview bid pair.</p>';return;
  }
  const query=new URLSearchParams({run_id:state.runId,step:state.selected,offset:state.page*20,limit:20});
  if(!baseline){query.set('a',state.cfA);query.set('b',state.cfB);}
  if(state.tokenQuery!=='')query.set('token_id',state.tokenQuery);
  showInspectPending();
  try{
    const result=await api(`/api/live/inspect?${query}`);
    if(epoch!==inspectEpoch||!mounted)return;
    state.inspect=result;paintInspection(result,e);
  }catch(error){if(epoch!==inspectEpoch||!mounted)return;$('#live-prob-table').innerHTML=`<p class="small">${esc(error.message)}</p>`;if($('#live-cf-outcome'))$('#live-cf-outcome').textContent=error.message;}
}

function paintInspection(result,event){
  const baseline=result.baseline,cf=result.counterfactual;
  const originalQ=row=>baseline?row.p:(event.bid_a*row.p_a+event.bid_b*row.p_b)/(event.bid_a+event.bid_b);
  const reference=Boolean(result.no_ad_comparison);
  const signedPp=p=>`${p>0?'+':''}${fmt(p*100,3)} pp`;
  const table=rows=>`<div class="table-wrap"><table class="probability-table"><thead><tr><th>Token / ID</th>${baseline?'<th>No-ad p</th>':'<th>p_A</th><th>p_B</th><th>Ad mixture q</th>'}${reference?'<th>No-ad p</th><th>q − no-ad</th>':''}${cf?'<th>Preview q</th>':''}</tr></thead><tbody>${rows.map(row=>`<tr class="${row.token_id===event.token_id?'chosen':''}"><td>${esc(row.token||'[empty]')} ${row.token_id===event.token_id?'←':''}<small class="token-number">#${row.token_id}</small></td>${baseline?`<td>${probability(row.p)}</td>`:`<td>${probability(row.p_a)}</td><td>${probability(row.p_b)}</td><td>${probability(originalQ(row))}</td>`}${reference?`<td>${probability(row.p_no_ad)}</td><td class="${row.delta_q_no_ad>=0?'prob-up':'prob-down'}">${signedPp(row.delta_q_no_ad)}</td>`:''}${cf?`<td>${probability(row.q)}</td>`:''}</tr>`).join('')}</tbody></table></div>`;
  const c=result.no_ad_comparison;
  $('#live-no-ad-shift').innerHTML=reference?`<div class="no-ad-shift"><p class="eyebrow">SAME PREFIX / PROMPT-ONLY REFERENCE</p><p class="small">No-ad p uses only your user prompt and the exact generated prefix shown on the right. The prefix can already contain ad content; it is held fixed for this comparison.</p><div class="shift-metrics">${[['A vs no-ad',c.tv_a],['B vs no-ad',c.tv_b],['Ad mixture vs no-ad',c.tv_q]].map(([label,v])=>`<div><span>${label}</span><strong>${probability(v)}</strong></div>`).join('')}</div><p class="small muted">Total variation across the full vocabulary: 0% = identical distributions; 100% = disjoint. Table changes are percentage points (pp). The no-ad reference never bids or affects sampling or payments.</p></div>`:baseline?'':'<p class="small muted">No-ad probabilities were not recorded for this earlier run. Refresh inference to include them.</p>';
  $('#live-pinned').innerHTML=`<p class="eyebrow">SELECTED TOKEN & RESIDUAL CANDIDATES</p>${table(result.pinned)}`;
  $('#live-prob-table').innerHTML=table(result.rows);
  $('#live-prob-mode').textContent=cf?'ACTUAL p / PREVIEW MIXTURE':'ACTUAL RUN';
  $('#live-page-label').textContent=state.tokenQuery!==''?`ID ${state.tokenQuery} / ${result.total.toLocaleString()} entries`:`${result.offset+1}–${Math.min(result.offset+result.limit,result.total)} of ${result.total.toLocaleString()}`;
  $('#live-page-prev').disabled=state.page===0||state.tokenQuery!=='';
  $('#live-page-next').disabled=(result.offset+result.limit>=result.total)||state.tokenQuery!=='';
  if(!baseline){
    const o=result.outcome;
    $('#live-cf-outcome').innerHTML=`<div class="outcome"><p class="eyebrow">${o.branch==='shared'?'SHARED BRANCH':o.branch+' WINS'} · ${cf?'COUNTERFACTUAL':'ORIGINAL BIDS'}</p><h3>${esc(o.token||'[empty]')}</h3><p>A pays ${fmt(o.payment_a,6)} · B pays ${fmt(o.payment_b,6)}</p></div>${event.branch==='shared'?'<p class="small muted section-gap">This shared-branch outcome stays the same for every valid bid pair.</p>':`<p class="small muted section-gap">With B = ${state.cfB}, A wins above ${fmt(state.cfB*(event.tape.contest/(1-event.tape.contest)),6)}. At the exact boundary, B wins. ${o.token_id===event.token_id?'The token matches':'The token differs from'} the actual selection.</p>`}`;
  }
}

function renderProvenance(){
  if(!mounted||!state.manifest)return;
  // Keep an opened prompt disclosure stable as new tokens arrive.
  if($('#live-provenance').dataset.run===state.runId){
    if(state.result&&!$('#live-result-link'))$('#live-files').insertAdjacentHTML('beforeend',`<span id="live-result-link">${fileLink('result.json','Final result')}</span>`);
    return;
  }
  $('#live-provenance').dataset.run=state.runId;
  const m=state.manifest;
  $('#live-provenance').innerHTML=`<div class="callout"><strong>Simulated charges, unedited output.</strong> Totals accumulate across tokens and can exceed a per-token bid. Passing the cache check establishes agreement between computation paths; it does not verify the generated claims.</div><details><summary>Exact prompts, generation settings, and run evidence</summary><div><p class="mono">${esc(state.runId)}</p><p class="small muted">${esc(m.model_id)} · ${esc(m.device)} / ${esc(m.dtype)} · seed ${m.config.seed} · temperature ${m.config.temperature} · cap ${m.config.max_tokens}. Model and tokenizer revision: <code>${esc(m.model_revision)}</code>.</p>${m.prompts.map((p,i)=>`<h3>${m.config.baseline?'No-ad context':`Advertiser ${i?'B':'A'}`} · ${p.token_ids.length} prompt tokens</h3><pre>${esc(p.text)}</pre>`).join('')}${m.no_ad_reference?`<h3>Prompt-only reference · ${m.no_ad_reference.token_ids.length} prompt tokens</h3><pre>${esc(m.no_ad_reference.text)}</pre><p class="small muted">Used both for a separate no-ad answer and for a read-only observer of the auction-selected prefix. The observer is not a bidder.</p>`:''}<div id="live-files">${fileLink('manifest.json','Full manifest')}${fileLink('events.jsonl','Original event trace')}${state.result?`<span id="live-result-link">${fileLink('result.json','Final result')}</span>`:''}</div><p class="small muted section-gap">Manifests preserve your inputs, rendered prompts, source hashes, pinned model/tokenizer revision, dependencies, seed, and generation settings. Local audit files are retained in <code>artifacts/live-lab/</code>. The interface exposes one active run; a new run replaces its probability inspector.</p></div></details>`;
}
