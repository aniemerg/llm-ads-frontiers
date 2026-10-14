import { mountLive, unmountLive } from './live.mjs';
import { mountDecode, unmountDecode, setDecodeText } from './decode.mjs';

function route(){
  unmountLive();unmountDecode();
  if(location.hash==='#decode')mountDecode();
  else mountLive();
}
window.addEventListener('hashchange',route);
window.addEventListener('decode-response',event=>{
  setDecodeText(event.detail);
  location.hash='decode';
});
route();
