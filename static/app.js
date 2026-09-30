/* Prediction transport can later be replaced with browser inference. */
const $ = id => document.getElementById(id);
const state = {source:'photo', stream:null, detecting:false, busy:false, timer:null, file:null, url:null, revision:0};
const labels = {good:'良品', bad:'瑕疵豆'};
function feedback(message='') { $('feedback').textContent=message; }
function resetResult() {
  document.querySelector('.result-panel').removeAttribute('data-result');
  $('result-symbol').textContent='—'; $('result-label').textContent='尚未辨識';
  $('result-description').textContent='選擇豆種與照片後，辨識結果會出現在這裡。';
  $('probability-section').hidden=true; $('status-badge').textContent='等待照片'; feedback();
}
function showResult(data) {
  const label=data.prediction.label;
  document.querySelector('.result-panel').dataset.result=label;
  $('result-symbol').textContent=label==='good'?'✓':'!'; $('result-label').textContent=labels[label]||label;
  $('result-description').textContent=label==='good'?'模型判斷這顆豆的外觀較接近良品。':'模型判斷這顆豆的外觀較接近瑕疵豆。';
  $('status-badge').textContent='辨識完成'; $('probability-section').hidden=false;
  $('last-update').textContent=new Date().toLocaleTimeString('zh-TW',{hour12:false});
  $('probability-container').replaceChildren();
  for(const item of data.probabilities) {
    const pct=Math.max(0,Math.min(100,item.probability*100));
    const row=document.createElement('div'); row.className='prob-item';
    const info=document.createElement('div'); info.className='prob-info';
    const name=document.createElement('span'); name.textContent=labels[item.label]||item.label;
    const value=document.createElement('span'); value.textContent=`${pct.toFixed(1)}%`; info.append(name,value);
    const track=document.createElement('div'); track.className='progress-bg';
    const bar=document.createElement('div'); bar.className='progress-fill'; bar.style.width=`${pct}%`;
    bar.style.background=item.label==='good'?'var(--success)':'var(--danger)';
    track.append(bar); row.append(info,track); $('probability-container').append(row);
  }
}
async function requestPrediction(file,modelKey,signal,progress) {
  if(window.browserInference) return window.browserInference.predict(file,modelKey,signal,progress);
  const body=new FormData(); body.append('image',file); body.append('model_key',modelKey);
  const response=await fetch('api/predict',{method:'POST',body,signal});
  const data=await response.json().catch(()=>({}));
  if(!response.ok) throw new Error(data.error||'辨識失敗，請稍後再試。');
  return data;
}
async function predict(file) {
  if(!file||!$('model-select').value||state.busy) return;
  const revision=state.revision; state.busy=true;
  $('status-badge').textContent='辨識中…'; feedback();
  const controller=new AbortController(); const timeout=setTimeout(()=>controller.abort(),window.browserInference?600000:45000);
  const progress=message=>{if(revision===state.revision) $('status-badge').textContent=message;};
  try { const data=await requestPrediction(file,$('model-select').value,controller.signal,progress); if(revision===state.revision) showResult(data); }
  catch(error) { if(revision===state.revision) { $('status-badge').textContent='未完成'; feedback(error.name==='AbortError'?'等候時間過長，請重新選擇豆種或照片再試。':error.message); stopDetection(); } }
  finally { clearTimeout(timeout); state.busy=false; if(revision!==state.revision&&state.source==='photo'&&state.file) predict(state.file); }
}
function stopDetection() {state.detecting=false; clearTimeout(state.timer); $('mode-toggle-btn').textContent='開始即時辨識';}
function stopCamera() {
  stopDetection(); state.stream?.getTracks().forEach(track=>track.stop()); state.stream=null;
  $('webcam-view').srcObject=null; $('webcam-view').hidden=true; $('camera-guide').hidden=true;
  $('camera-toggle-btn').textContent='開啟相機'; $('mode-toggle-btn').disabled=true;
}
function setSource(source) {
  state.revision++; stopCamera(); state.source=source; resetResult();
  for(const mode of ['photo','camera']) { $(`${mode}-tab`).classList.toggle('active',source===mode); $(`${mode}-tab`).setAttribute('aria-pressed',String(source===mode)); }
  $('camera-controls').hidden=source!=='camera';
  const hasPhoto=source==='photo'&&Boolean(state.file);
  $('static-preview').hidden=!hasPhoto; $('replace-btn').hidden=!hasPhoto; $('visual-placeholder').hidden=hasPhoto;
  $('visual-placeholder').querySelector('h3').textContent=source==='photo'?'把照片放在這裡':'相機辨識';
  $('visual-placeholder').querySelector('p').textContent=source==='photo'?'拖曳照片，或選擇裝置裡的圖片':'開啟相機，將單顆咖啡豆放在中央';
  $('upload-btn').hidden=source==='camera'; document.querySelector('.file-hint').hidden=source==='camera';
  if(hasPhoto) predict(state.file);
}
function usePhoto(file) {
  if(!file) return;
  if(!['image/jpeg','image/png','image/webp','image/bmp'].includes(file.type)) {feedback('請使用 JPG、PNG、WEBP 或 BMP 圖片。'); return;}
  if(file.size>10*1024*1024) {feedback('照片超過 10 MB，請縮小後再試。'); return;}
  state.file=file; if(state.url) URL.revokeObjectURL(state.url);
  state.url=URL.createObjectURL(file); $('static-preview').src=state.url; setSource('photo');
  if(!$('model-select').value) feedback('照片已準備好，請先選擇咖啡豆種類。');
}
async function startCamera() {
  if(state.stream) {state.revision++; stopCamera(); $('visual-placeholder').hidden=false; resetResult(); return;}
  const revision=state.revision; $('camera-toggle-btn').disabled=true;
  try {
    if(!navigator.mediaDevices?.getUserMedia) throw new Error('相機不支援');
    const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:640},height:{ideal:640}},audio:false});
    if(revision!==state.revision||state.source!=='camera') {stream.getTracks().forEach(track=>track.stop()); return;}
    state.stream=stream; $('webcam-view').srcObject=stream; await $('webcam-view').play();
    if(revision!==state.revision||state.source!=='camera') {stream.getTracks().forEach(track=>track.stop()); return;}
    $('webcam-view').hidden=false; $('visual-placeholder').hidden=true; $('camera-guide').hidden=false;
    $('camera-toggle-btn').textContent='關閉相機'; $('mode-toggle-btn').disabled=!$('model-select').value;
    feedback($('model-select').value?'':'相機已開啟，請選擇咖啡豆種類。');
  } catch(error) {if(revision===state.revision) {stopCamera(); feedback(error.name==='NotAllowedError'?'請允許相機存取，或改用照片上傳。':'無法開啟相機，請確認裝置與瀏覽器設定，或改用照片。');}}
  finally {$('camera-toggle-btn').disabled=false;}
}
async function cameraLoop() {
  if(!state.detecting||!state.stream) return;
  const revision=state.revision, video=$('webcam-view');
  if(!state.busy&&video.readyState>=2&&video.videoWidth) {
    const canvas=$('frame-capture'), scale=Math.min(1,640/Math.max(video.videoWidth,video.videoHeight));
    canvas.width=Math.round(video.videoWidth*scale); canvas.height=Math.round(video.videoHeight*scale);
    canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.85));
    if(revision===state.revision&&state.detecting&&blob) await predict(new File([blob],'camera.jpg',{type:'image/jpeg'}));
  }
  if(revision===state.revision&&state.detecting) state.timer=setTimeout(cameraLoop,500);
}
$('upload-btn').addEventListener('click',()=>$('image-input').click());
$('replace-btn').addEventListener('click',()=>$('image-input').click());
$('image-input').addEventListener('change',event=>{usePhoto(event.target.files[0]); event.target.value='';});
$('photo-tab').addEventListener('click',()=>setSource('photo'));
$('camera-tab').addEventListener('click',()=>setSource('camera'));
$('camera-toggle-btn').addEventListener('click',startCamera);
$('mode-toggle-btn').addEventListener('click',()=>{
  if(state.detecting) {state.revision++; stopDetection();}
  else if(state.stream&&$('model-select').value) {state.detecting=true; $('mode-toggle-btn').textContent='暫停辨識'; cameraLoop();}
});
$('model-select').addEventListener('change',()=>{
  state.revision++; resetResult(); const option=$('model-select').selectedOptions[0];
  $('selected-bean').textContent=option.textContent; $('info-arch').textContent=option.dataset.arch;
  $('info-size').textContent=`${option.dataset.size} × ${option.dataset.size}`; $('mode-toggle-btn').disabled=!state.stream;
  if(state.source==='photo') predict(state.file); else stopDetection();
});
for(const type of ['dragenter','dragover','dragleave','drop']) $('drop-zone').addEventListener(type,event=>{
  event.preventDefault(); $('drop-zone').classList.toggle('drag-over',type==='dragenter'||type==='dragover');
  if(type==='drop') usePhoto(event.dataTransfer.files[0]);
});
let exampleRevision=0;
document.querySelectorAll('.example-thumb').forEach(button=>button.addEventListener('click',async()=>{
  const revision=++exampleRevision;
  try {const response=await fetch(button.dataset.src); if(!response.ok) throw new Error('範例照片載入失敗，請稍後再試。');
    const blob=await response.blob(); if(revision!==exampleRevision) return;
    document.querySelectorAll('.example-thumb').forEach(item=>item.classList.toggle('active',item===button));
    usePhoto(new File([blob],'example.jpg',{type:'image/jpeg'}));
  }catch(error){feedback(error.message);}
}));
window.addEventListener('pagehide',stopCamera);
