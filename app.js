let comps=[],eps=[],removed=new Set(),fileURLs={},fpUrl=null;
let curEp=null,playing=false,autoPlay=true,hist=[];
const audio=document.createElement('audio');audio.preload='auto';
let progressRAF=null,dbgLines=[];
function log(m){dbgLines.push(m);if(dbgLines.length>200)dbgLines.shift();const el=document.getElementById('dbg');if(el){el.style.display='';el.textContent=dbgLines.join('\n');el.scrollTop=9e9}}
function save(){try{localStorage.setItem('km',JSON.stringify({comps,eps,removed:[...removed]}))}catch(e){}}
function load(){try{const d=JSON.parse(localStorage.getItem('km'));if(d){comps=d.comps||[];eps=d.eps||[];removed=new Set(d.removed||[])}}catch(e){}}
function fmt(s){if(!isFinite(s)||s<0)return'0:00';const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60);return h>0?h+':'+String(m).padStart(2,'0')+':'+String(sec).padStart(2,'0'):m+':'+String(sec).padStart(2,'0')}

// ===================== JINGLE DETECTION =====================
// Find the sharpest quiet→loud energy transition in each search window.
// ratio = max(next 400ms) / avg(previous 800ms)
// The jingle horn blast after inter-episode silence always produces
// the highest ratio in its 3-4 min search window.
// Split placed 1s before the detected horn (in the silence gap).

function findMp3Sync(bytes, off) {
  for (let i = Math.max(0,off); i < bytes.length - 3; i++) {
    if (bytes[i]!==0xFF||(bytes[i+1]&0xE0)!==0xE0) continue;
    const v=(bytes[i+1]>>3)&3,l=(bytes[i+1]>>1)&3,br=(bytes[i+2]>>4)&0xF,sr=(bytes[i+2]>>2)&3;
    if(v!==1&&l!==0&&br>0&&br<15&&sr<3) return i;
  }
  return -1;
}

function findJingleInSlice(pcm, sr) {
  const frameSz = Math.round(sr * 0.05);
  const energy = [];
  for (let i = 0; i < pcm.length; i += frameSz) {
    const end = Math.min(i + frameSz, pcm.length);
    let sum = 0;
    for (let j = i; j < end; j++) sum += pcm[j]*pcm[j];
    energy.push(Math.sqrt(sum/(end-i)));
  }
  if (energy.length < 30) return -1;

  const NB = 16;       // 800ms lookback
  const NA = 8;        // 400ms lookahead
  const FLOOR = 0.01;  // minimum denominator
  const MIN_RATIO = 2.0;

  let bestRatio = 0, bestFrame = -1;
  for (let i = NB; i < energy.length - NA; i++) {
    let bSum = 0;
    for (let j = i - NB; j < i; j++) bSum += energy[j];
    const before = bSum / NB;
    let after = 0;
    for (let k = i; k < i + NA; k++) {
      if (energy[k] > after) after = energy[k];
    }
    const ratio = after / Math.max(before, FLOOR);
    if (ratio > bestRatio) { bestRatio = ratio; bestFrame = i; }
  }

  if (bestFrame < 0 || bestRatio < MIN_RATIO) return -1;
  return Math.max(0, bestFrame * 0.05 - 1.0);
}

async function analyzeFile(url, fileAB, onProgress, onStatus) {
  const bytes = new Uint8Array(fileAB);
  const totalBytes = bytes.length;
  const tmpA = new Audio(); tmpA.src = url; tmpA.preload = 'metadata';
  const dur = await new Promise((ok,no) => {
    tmpA.onloadedmetadata = () => ok(tmpA.duration);
    tmpA.onerror = () => no(new Error('Format non supporté'));
    setTimeout(() => no(new Error('Timeout')), 20000);
  });
  tmpA.src = '';
  log('Durée: ' + fmt(dur) + ' (' + (totalBytes/1048576).toFixed(1) + ' MB)');
  const bps = totalBytes / dur;
  const SLICE_SEC = 100;
  const sliceBytes = Math.round(SLICE_SEC * bps);
  let ctx;
  try { ctx = new AudioContext({sampleRate:8000}); } catch(e) {
    try { ctx = new AudioContext({sampleRate:22050}); } catch(e2) { ctx = new AudioContext(); }
  }
  log('SR: ' + ctx.sampleRate);
  const jingles = [0];
  let lastJ = 0;
  let consecutiveFails = 0;
  for (let epIdx = 0; epIdx < Math.ceil(dur/150); epIdx++) {
    const searchStart = lastJ + 155;
    const searchEnd = Math.min(lastJ + 245, dur - 5);
    if (searchStart >= dur - 10) break;
    const sliceStartSec = Math.max(0, searchStart - 5);
    const sliceStartByte = findMp3Sync(bytes, Math.round(sliceStartSec * bps));
    if (sliceStartByte < 0) break;
    const sliceEndByte = Math.min(totalBytes, sliceStartByte + sliceBytes + 5000);
    const slice = fileAB.slice(sliceStartByte, sliceEndByte);
    onStatus('Jingle ' + (epIdx+1) + ' — ' + fmt(searchStart));
    onProgress(searchStart / dur);
    try {
      const buf = await ctx.decodeAudioData(slice.slice(0));
      const pcm = buf.getChannelData(0);
      const actualSliceStart = sliceStartByte / bps;
      const hitInSlice = findJingleInSlice(pcm, ctx.sampleRate);
      if (hitInSlice >= 0) {
        const absTime = actualSliceStart + hitInSlice;
        if (absTime > lastJ + 140 && absTime < lastJ + 260) {
          jingles.push(absTime);
          log('♪ #'+jingles.length+' '+fmt(absTime)+' ep='+fmt(absTime-lastJ));
          lastJ = absTime;
          consecutiveFails = 0;
          continue;
        }
      }
    } catch(e) { log('⚠ Decode: ' + e.message); }
    consecutiveFails++;
    if (consecutiveFails >= 3) { log('✖ 3 fails, stop'); break; }
    const fb = lastJ + 200;
    if (fb >= dur - 10) break;
    jingles.push(fb);
    log('⚠ Fallback ' + fmt(fb));
    lastJ = fb;
  }
  try { ctx.close(); } catch(e) {}
  const splits = [...jingles, dur];
  const ds=[]; for(let i=0;i<splits.length-1;i++) ds.push(splits[i+1]-splits[i]);
  if(ds.length) log('→ '+ds.length+' ép, moy='+fmt(ds.reduce((a,b)=>a+b,0)/ds.length));
  return { splitPoints: splits, totalDuration: dur };
}

// ===================== PLAYBACK =====================
function stopPlay(){if(progressRAF){cancelAnimationFrame(progressRAF);progressRAF=null}audio.pause();playing=false;render()}
function playEp(ep){stopPlay();const url=fileURLs[ep.compilationId];if(!url)return;curEp=ep;if(audio.getAttribute('data-cid')!==ep.compilationId){audio.src=url;audio.setAttribute('data-cid',ep.compilationId)}const go=()=>{audio.currentTime=ep.startSec;const p=()=>{audio.play().then(()=>{playing=true;hist=[ep,...hist.filter(h=>h.id!==ep.id).slice(0,19)];render();startPL();updateMS()}).catch(e=>log('ERR:'+e.message))};if(audio.seeking)audio.addEventListener('seeked',p,{once:true});else p()};if(audio.readyState>=1)go();else{audio.addEventListener('loadedmetadata',go,{once:true});audio.load()}}
function startPL(){const tick=()=>{if(!playing||!curEp)return;if(audio.currentTime>=curEp.endSec-.05){const f=curEp;stopPlay();if(autoPlay)setTimeout(()=>playRand(f.id),500);return}const el=Math.max(0,audio.currentTime-curEp.startSec),pct=Math.min(100,el/curEp.duration*100);const pf=document.getElementById('pFill');if(pf)pf.style.width=pct+'%';const mf=document.getElementById('mpFill');if(mf)mf.style.width=pct+'%';const te=document.getElementById('tE');if(te)te.textContent=fmt(el);progressRAF=requestAnimationFrame(tick)};progressRAF=requestAnimationFrame(tick)}
function playRand(exId){const a=eps.filter(e=>!removed.has(e.id)&&e.id!==exId);if(!a.length)return;playEp(a[Math.floor(Math.random()*a.length)])}
function togPause(){if(!curEp){playRand();return}if(playing){audio.pause();if(progressRAF){cancelAnimationFrame(progressRAF);progressRAF=null}playing=false}else{audio.play().then(()=>{playing=true;startPL()}).catch(e=>log('ERR:'+e.message))}render()}

async function handleFiles(files){const ldA=document.getElementById('ldArea'),ldT=document.getElementById('ldTxt'),ldS=document.getElementById('ldSub'),ldF=document.getElementById('ldFill');ldA.style.display='';document.getElementById('uplArea').style.display='none';
for(let fi=0;fi<files.length;fi++){const file=files[fi];ldT.textContent='Analyse '+(fi+1)+'/'+files.length;ldS.textContent=file.name;ldF.style.width='0%';log('━━━ '+file.name+' ━━━');
try{const url=URL.createObjectURL(file);const cid='c'+Date.now()+'_'+fi;fileURLs[cid]=url;const ab=await file.arrayBuffer();fpUrl=url;
const res=await analyzeFile(url,ab,p=>{ldF.style.width=Math.round(p*100)+'%'},s=>{ldS.textContent=s});
ldF.style.width='100%';ldS.textContent=(res.splitPoints.length-1)+' épisodes !';
const nm=file.name.replace(/\.[^.]+$/,'');comps.push({id:cid,name:nm,totalDuration:res.totalDuration});
for(let i=0;i<res.splitPoints.length-1;i++){const s=res.splitPoints[i],e=res.splitPoints[i+1];if(e-s<5)continue;eps.push({id:cid+'_e'+i,compilationId:cid,compilationName:nm,index:i,startSec:s,endSec:e,duration:e-s,label:'Épisode '+(i+1)})}
document.getElementById('fpArea').style.display='';
await new Promise(r=>setTimeout(r,300))}catch(err){log('ERREUR: '+err.message);alert('Erreur: '+err.message)}}save();ldA.style.display='none';render()}

document.getElementById('bListenFP').onclick=()=>{if(!fpUrl)return;audio.src=fpUrl;audio.setAttribute('data-cid','');audio.currentTime=0;audio.play();setTimeout(()=>audio.pause(),2000)};
document.getElementById('bCopyDbg').onclick=()=>{navigator.clipboard.writeText(dbgLines.join('\n')).then(()=>{document.getElementById('bCopyDbg').textContent='✓ Copié !';setTimeout(()=>document.getElementById('bCopyDbg').textContent='📋 Copier log',2000)}).catch(()=>{prompt('Log:',dbgLines.join('\n'))})};

function togRm(id){if(removed.has(id))removed.delete(id);else removed.add(id);save();render()}
function rmComp(id){if(!confirm('Supprimer ?'))return;if(curEp?.compilationId===id)stopPlay();comps=comps.filter(c=>c.id!==id);eps=eps.filter(e=>e.compilationId!==id);if(fileURLs[id]){URL.revokeObjectURL(fileURLs[id]);delete fileURLs[id]}save();render()}
function resetAll(){if(!confirm('Réinitialiser ?'))return;stopPlay();Object.values(fileURLs).forEach(u=>URL.revokeObjectURL(u));comps=[];eps=[];removed.clear();fileURLs={};hist=[];curEp=null;dbgLines=[];fpUrl=null;document.getElementById('fpArea').style.display='none';localStorage.removeItem('km');render()}

function render(){const has=eps.length>0,act=eps.filter(e=>!removed.has(e.id)),rmE=eps.filter(e=>removed.has(e.id));document.getElementById('uplArea').style.display=has?'none':'';document.getElementById('plContent').style.display=has?'':'none';document.getElementById('sA').textContent=act.length;document.getElementById('sR').textContent=removed.size;document.getElementById('sC').textContent=comps.length;const npc=document.getElementById('npCard');if(curEp){npc.style.display='';document.getElementById('npT').textContent=curEp.label;document.getElementById('npS').textContent=curEp.compilationName;document.getElementById('tT').textContent=fmt(curEp.duration)}else npc.style.display='none';document.getElementById('bPlay').textContent=playing?'⏸':'▶';document.getElementById('mpBtn').textContent=playing?'⏸':'▶';document.getElementById('bAuto').classList.toggle('on',autoPlay);const mp=document.getElementById('miniP');if(curEp){mp.classList.add('vis');document.getElementById('mpT').textContent=curEp.label;document.getElementById('mpS').textContent=curEp.compilationName}else mp.classList.remove('vis');if(hist.length){document.getElementById('histArea').style.display='';const hl=document.getElementById('histList');hl.innerHTML='';hist.slice(0,5).forEach((ep,i)=>{const d=document.createElement('div');d.className='hi'+(i===0&&playing?' pl':'');d.innerHTML='<span class="hn">'+ep.label+'</span><span class="hd">'+fmt(ep.duration)+'</span>';d.onclick=()=>playEp(ep);hl.appendChild(d)})}document.getElementById('epCnt').textContent=act.length+' ÉPISODES DISPONIBLES';const el=document.getElementById('epList');el.innerHTML='';if(!act.length){document.getElementById('epEy').style.display='';document.getElementById('epEyM').textContent=has?'Tous exclus.':'Chargez des compilations.'}else{document.getElementById('epEy').style.display='none';const grp={};act.forEach(e=>{(grp[e.compilationId]=grp[e.compilationId]||[]).push(e)});for(const[cid,list]of Object.entries(grp)){const cp=comps.find(c=>c.id===cid);const h=document.createElement('div');h.className='ch';h.textContent='🏰 '+(cp?cp.name:cid);el.appendChild(h);list.sort((a,b)=>a.index-b.index).forEach(ep=>{const d=document.createElement('div');d.className='ei'+(curEp?.id===ep.id?' pl':'');d.innerHTML='<button class="ep" data-play="'+ep.id+'">▶</button><div class="eif"><div class="en">'+ep.label+'</div><div class="em">'+fmt(ep.startSec)+' → '+fmt(ep.endSec)+' · '+fmt(ep.duration)+'</div></div><button class="er" data-rm="'+ep.id+'">✕</button>';el.appendChild(d)})}}document.getElementById('cpH').textContent='COMPILATIONS ('+comps.length+')';const cl=document.getElementById('cpL');cl.innerHTML='';document.getElementById('noCp').style.display=comps.length?'none':'';comps.forEach(cp=>{const nn=eps.filter(e=>e.compilationId===cp.id).length;const d=document.createElement('div');d.className='cc';d.innerHTML='<div class="cci"><div class="ccn">'+cp.name+'</div><div class="ccm">'+nn+' épisodes · '+fmt(cp.totalDuration)+'</div></div><button class="bds" data-del="'+cp.id+'">Supprimer</button>';cl.appendChild(d)});if(rmE.length){document.getElementById('rmSec').style.display='';document.getElementById('rmH').textContent='ÉPISODES EXCLUS ('+rmE.length+')';const rl=document.getElementById('rmL');rl.innerHTML='';rmE.forEach(ep=>{const d=document.createElement('div');d.className='ri';d.innerHTML='<span class="rin">'+ep.label+' — '+ep.compilationName+'</span><button class="ers" data-rest="'+ep.id+'">Restaurer</button>';rl.appendChild(d)})}else document.getElementById('rmSec').style.display='none'}

function updateMS(){if(!('mediaSession'in navigator)||!curEp)return;navigator.mediaSession.metadata=new MediaMetadata({title:curEp.label,artist:'Kaamelott',album:curEp.compilationName});navigator.mediaSession.setActionHandler('play',togPause);navigator.mediaSession.setActionHandler('pause',togPause);navigator.mediaSession.setActionHandler('nexttrack',()=>playRand(curEp?.id))}
document.addEventListener('click',e=>{const t=e.target;if(t.dataset.play){const ep=eps.find(x=>x.id===t.dataset.play);if(ep)playEp(ep)}if(t.dataset.rm)togRm(t.dataset.rm);if(t.dataset.del)rmComp(t.dataset.del);if(t.dataset.rest)togRm(t.dataset.rest)});
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));document.querySelectorAll('.pnl').forEach(p=>p.classList.remove('on'));tab.classList.add('on');document.getElementById('p-'+tab.dataset.t).classList.add('on')}));
document.getElementById('fIn').addEventListener('change',e=>{if(e.target.files.length)handleFiles(e.target.files);e.target.value=''});
document.getElementById('bLoad').onclick=()=>document.getElementById('fIn').click();
document.getElementById('bMore').onclick=()=>document.getElementById('fIn').click();
document.getElementById('bPlay').onclick=togPause;
document.getElementById('bStop').onclick=()=>{stopPlay();curEp=null;render()};
document.getElementById('bAuto').onclick=()=>{autoPlay=!autoPlay;render()};
document.getElementById('bRestAll').onclick=()=>{removed.clear();save();render()};
document.getElementById('bReset').onclick=resetAll;
document.getElementById('mpBtn').onclick=togPause;
document.getElementById('mpSh').onclick=()=>playRand(curEp?.id);
audio.onerror=()=>log('ERR: '+(audio.error?.message||'?'));
load();render();
