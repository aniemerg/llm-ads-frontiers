const $=selector=>document.querySelector(selector);
const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state={text:'',result:null,busy:false,error:'',examples:[],note:'',source:null};
let mounted=false,epoch=0;

export function setDecodeText(text){
  epoch++;state.busy=false;
  state.text=text;state.result=null;state.error='';state.note='Exact text copied from the generator. Only text is sent to the decoder.';state.source=null;
}
export function unmountDecode(){mounted=false;}
export async function mountDecode(){
  mounted=true;render();
  if(!state.examples.length){
    try{
      const response=await fetch('/api/watermark/examples');
      if(!response.ok)throw new Error('Recorded examples are unavailable.');
      state.examples=await response.json();
      if(mounted)renderExamples();
    }catch(error){state.error=error.message;if(mounted)renderResult();}
  }
}

function renderExamples(){
  const select=$('#decode-example');
  if(!select)return;
  select.innerHTML='<option value="">Load a recorded example…</option>'+state.examples.map((e,i)=>`<option value="${i}">${esc(e.label)}</option>`).join('');
}

function render(){
  if(!mounted)return;
  $('#content').innerHTML=`<div class="live-toolbar"><h1>Ads in LLMs.</h1><nav class="app-nav" aria-label="Pages"><a href="#live">Generate</a><a href="#decode" aria-current="page">Decode</a></nav></div><section class="panel decode-input"><h2>Decode watermark</h2><p class="small muted">Paste the exact response. Decoding uses only this text and the fixed experimental protocol.</p><fieldset id="decode-fields"><label for="decode-text">Response text</label><textarea id="decode-text" rows="10" maxlength="60000" spellcheck="false">${esc(state.text)}</textarea><div class="controls decode-actions"><button id="decode-submit" class="primary">Decode</button><select id="decode-example" aria-label="Recorded example"></select><span id="decode-progress" role="status"></span></div></fieldset><p id="decode-note" class="small muted">${esc(state.note)}</p><details><summary>Protocol and limitations</summary><p class="small muted">Qwen3-4B-Instruct-2507 · tokenizer revision cdbee75f17c01a7cc42f958dc650907174af0554 · tilt + race v1 · public key 1901 · vocabulary 151,936 · 64-token blocks · spans up to 32 tokens · limit 4,096 tokens · nominal α = 0.01.</p><p class="small muted">The earliest complete carrier ends at token 128. Short, edited, cropped, or differently configured text may not decode. Carried continuation tokens retain their positions and can weaken recovery; combined-mode recovery and false positives are not calibrated. One of six unmarked research controls was falsely flagged. Statistical detection is not authentication, proof of payment, or a factual accuracy check.</p></details></section><div id="decode-results" aria-live="polite"></div>`;
  renderExamples();
  $('#decode-text').addEventListener('input',()=>{
    state.text=$('#decode-text').value;state.result=null;state.error='';state.note='';state.source=null;
    $('#decode-note').textContent='';renderResult();
  });
  $('#decode-submit').addEventListener('click',runDecode);
  $('#decode-example').addEventListener('change',()=>{
    if($('#decode-example').value==='')return;
    const example=state.examples[Number($('#decode-example').value)];
    state.text=example.text;state.note=example.note;state.source=example.source;
    state.result=null;state.error='';$('#decode-text').value=state.text;
    $('#decode-note').textContent=state.note;renderResult();
  });
  renderResult();
}

function highlighted(text,spans){
  // Partition by all boundaries, so overlapping detections never duplicate text.
  const edges=[...new Set([0,text.length,...spans.flatMap(s=>[s.start_utf16,s.end_utf16])])].sort((a,b)=>a-b);
  return edges.slice(0,-1).map((lo,i)=>{
    const hi=edges[i+1],matches=spans.filter(s=>s.start_utf16<hi&&s.end_utf16>lo);
    const fragment=esc(text.slice(lo,hi));
    return matches.length?`<mark class="watermark-highlight" title="${esc(matches.map(s=>`[${s.start_token}, ${s.end_token})`).join(', '))}">${fragment}</mark>`:fragment;
  }).join('');
}

function renderResult(){
  if(!mounted)return;
  $('#decode-fields').disabled=state.busy;
  $('#decode-progress').innerHTML=state.busy?'<span class="spinner"></span> Decoding…':'';
  const target=$('#decode-results'),r=state.result;
  if(state.error){target.innerHTML=`<div class="callout warning">${esc(state.error)}</div>`;return;}
  if(!r){target.innerHTML='';return;}
  const titles={detected:'Detected candidates',not_detected:'No confident match',ambiguous:'Ambiguous evidence',insufficient_evidence:'Insufficient text',unsupported_length:'Response is too long',unsupported_text:'Unsupported text'};
  const explanation=r.status==='insufficient_evidence'?`This response has ${r.token_count} tokens. The protocol needs at least 128, with an earlier mention and a complete later carrier block.`:r.status==='unsupported_length'?'The limit is 4,096 tokens. The text was not truncated or decoded.':r.status==='detected'?'These spans passed the experimental detector. False positives and missed mentions are possible.':'No unambiguous watermark was recovered. This does not establish that the response contains no advertising.';
  target.innerHTML=`<section class="panel decode-output"><div class="replay-head"><h2>${titles[r.status]||esc(r.status)}</h2><button id="decode-export">Export JSON</button></div><p>${explanation}</p><p class="small muted">Coverage unknown · ${r.token_count} tokens · ${r.spans.length} detected spans${state.source?' · recorded text replay':''}</p><div class="output-text decoded-text">${highlighted(state.text,r.spans)}</div>${r.spans.map((s,i)=>`<details class="decoded-span"><summary><span class="watermark-swatch"></span>${esc(s.literal_text)} <span class="mono">[${s.start_token}, ${s.end_token})</span></summary><p class="small">Zero-based token range; the end is excluded. Leading spaces may belong to the span.</p><p class="small">Evidence carried in later tokens [${s.carrier_start}, ${s.carrier_end}). Score ${s.score.toFixed(3)}; nominal corrected p-value ${s.nominal_corrected_p.toExponential(3)}. This is not a calibrated confidence percentage.</p></details>`).join('')}<details><summary>Full decoding evidence</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details></section>`;
  $('#decode-export').addEventListener('click',()=>{
    const blob=new Blob([JSON.stringify({text:state.text,...r,recorded_source:state.source},null,2)+'\n'],{type:'application/json'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download='watermark-decoded.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
}

async function runDecode(){
  if(state.busy)return;
  if(!state.text){state.error='Paste a response first.';renderResult();return;}
  const requestEpoch=++epoch;
  state.busy=true;state.error='';state.result=null;renderResult();
  try{
    const response=await fetch('/api/watermark/decode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:state.text})});
    const value=await response.json();
    if(!response.ok)throw new Error(value.error||'Decoding failed.');
    if(requestEpoch===epoch)state.result=value;
  }catch(error){if(requestEpoch===epoch)state.error=error.message;}
  finally{if(requestEpoch===epoch){state.busy=false;renderResult();}}
}
