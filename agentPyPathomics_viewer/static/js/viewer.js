let slideName = null;
let mpp = null; // microns per pixel
let objective = null; // objective magnification if provided by slide metadata (e.g., 20, 40)

// --- Overlay: nuclei_seg (semantic class) ---
const NUCLEI_SEG_DOWNSAMPLE = 16; // server will stream-downsample mask to this factor
let overlaysMeta = {};
let nucleiSegAvailable = false;
let nucleiSegDziAvailable = false;
let nucleiSegEnabled = false;
let nucleiSegOpacity = 0.90;
let nucleiSegItem = null;
let nucleiSegLoading = false;
let nucleiSegItems = [];

// --- Overlay: tissue_mask (tissue region mask) ---
const TISSUE_MASK_DOWNSAMPLE = 16;
let tissueMaskAvailable = false;
let tissueMaskEnabled = false;
let tissueMaskOpacity = 0.90;
let tissueMaskItem = null;
let tissueMaskLoading = false;
let tissueMaskItems = [];
let tissueMaskColorTag = null;

// --- Overlay: tissue_seg/full_images (RGB visualization) ---
let tissueSegFullAvailable = false;
let tissueSegFullEnabled = false;
let tissueSegFullOpacity = 0.90;
let tissueSegFullItem = null;
let tissueSegFullLoading = false;
let tissueSegFullItems = [];

// Diagnostics panel for quick status (helps when DevTools not available)
function createDiagPanel(){
  if(document.getElementById('diag')) return;
  const d = document.createElement('div'); d.id='diag';
  d.style.position='fixed'; d.style.left='10px'; d.style.bottom='10px'; d.style.zIndex=99999;
  d.style.background='rgba(0,0,0,0.8)'; d.style.color='#fff'; d.style.padding='8px'; d.style.fontSize='12px'; d.style.maxWidth='40vw'; d.style.maxHeight='30vh'; d.style.overflow='auto'; d.style.borderRadius='6px';
  d.innerHTML = '<b>Viewer diag</b><div id="diag-messages" style="margin-top:6px"></div>';
  document.body.appendChild(d);
}
function diag(msg){
  console.log('[diag]', msg);
  try{
    createDiagPanel();
    const el = document.getElementById('diag-messages');
    const p = document.createElement('div'); p.textContent = msg; el.appendChild(p);
    // keep only last 30
    while(el.childNodes.length > 30) el.removeChild(el.firstChild);
  }catch(e){/* ignore */}
}
// initialize small diag marker
try{ diag('viewer.js loaded'); }catch(e){}

const slidesSelect = document.getElementById('slides');
const reloadBtn = document.getElementById('reload');
const fileInput = document.getElementById('fileinput');

function fetchSlides(){
  diag('fetchSlides()');
  fetch('/slides').then(r=>r.json()).then(list=>{
    diag('slides fetched: ' + list.length);
    slidesSelect.innerHTML = '';
    list.forEach(fn => {
      const opt = document.createElement('option'); opt.value = fn; opt.textContent = fn; slidesSelect.appendChild(opt);
    });
    if(list.length) { slidesSelect.value = list[0]; setSlide(list[0]); }
    renderManage(list);
  }).catch(e=>{ diag('fetchSlides error: '+e); console.error(e); });
}

function setSlide(name){
  diag('[viewer] setSlide ' + name);
  console.log('[viewer] setSlide', name);
  slideName = name;
  updateManageListActive(name);
  if(!name) return;
  fetch(`/metadata/${encodeURIComponent(name)}`).then(r=>r.json()).then(meta=>{
    mpp = meta.mpp;
    objective = meta.objective || null;
    window.slidePyramid = meta.pyramid || null;
    overlaysMeta = meta.overlays || {};
    nucleiSegDziAvailable = !!overlaysMeta.nuclei_seg_dzi;
    // Consider nuclei overlay available if either downsampled preview exists OR DZI exists.
    // This allows running with only nuclei_seg_dzi copied (no huge nuclei_seg/*_class.png needed).
    nucleiSegAvailable = !!(overlaysMeta.nuclei_seg || overlaysMeta.nuclei_seg_dzi);
    tissueMaskAvailable = !!overlaysMeta.tissue_mask;
    tissueSegFullAvailable = !!overlaysMeta.tissue_seg_full;
    try{
      const om = meta.overlay_meta || {};
      tissueMaskColorTag = om.tissue_mask_color_tag || null;
    }catch(e){ tissueMaskColorTag = null; }
    updateOverlayAvailabilityUI();
    diag('[viewer] metadata: width='+meta.width+' height='+meta.height+' mpp='+meta.mpp+' pyramid='+ (meta.pyramid?meta.pyramid.length:0) + ' nuclei_seg=' + nucleiSegAvailable + ' nuclei_seg_dzi=' + nucleiSegDziAvailable + ' tissue_mask=' + tissueMaskAvailable + ' tissue_seg_full=' + tissueSegFullAvailable);
    console.log('[viewer] metadata', meta);
    initViewer(name);
  }).catch(err=>{ diag('[viewer] metadata fetch error: '+err); console.error('[viewer] metadata fetch error', err); });
}

let viewer = null;
// Expose viewer globally for layout functions to access redraw
window.viewer = null;
function initViewer(name){
  diag('[viewer] initViewer ' + name);
  console.log('[viewer] initViewer', name);
  try{
    // reset overlay references (new viewer instance)
    nucleiSegItem = null;
    nucleiSegLoading = false;
    nucleiSegItems = [];
    tissueMaskItem = null;
    tissueMaskLoading = false;
    tissueMaskItems = [];
    tissueSegFullItem = null;
    tissueSegFullLoading = false;
    tissueSegFullItems = [];
    if(viewer) { try{ viewer.destroy(); }catch(e){ console.warn('[viewer] destroy failed', e); } }
    const ts = `/dzi/${encodeURIComponent(name)}.dzi`;
    diag('[viewer] creating OpenSeadragon viewer, tileSources=' + ts);
    console.log('[viewer] creating OpenSeadragon viewer, tileSources=', ts);
    viewer = OpenSeadragon({
      id: 'viewer',
      prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
      tileSources: ts,
      showNavigator: false,
      // Hide built-in navigation buttons (+ / - / home etc.)
      showNavigationControl: false,
      showZoomControl: false,
      showHomeControl: false,
      // hide full-page toggle per request
      showFullPageControl: false,
      gestureSettingsMouse: { scrollToZoom: true, clickToZoom: false },
    });
    window.viewer = viewer;

    // ensure overlay is attached to the new viewer and resize/view handlers keep it updated
    viewer.addHandler('open', ()=>{
      diag('[viewer] open event');
      console.log('[viewer] open event, items=', viewer.world && viewer.world.getItemCount ? viewer.world.getItemCount() : 'n/a');

      // Move scalebar into viewer container so it's positioned bottom-left of the viewer
      try{
        const sb = document.getElementById('scalebar');
        if(sb && viewer && viewer.container && sb.parentElement !== viewer.container){
          viewer.container.appendChild(sb);
          sb.style.position = 'absolute'; sb.style.left = '20px'; sb.style.bottom = '20px'; sb.style.zIndex = 2000;
          diag('[scalebar] moved into viewer container');
        }
      }catch(e){ console.warn('[scalebar] move failed', e); }

      updateScaleBar(); updateZoomLabel();
      updateViewerLayout();

      // Create/update pyramid level quick-jump controls
      try{ updatePyramidControls(); }catch(e){ console.warn('[pyramid] update failed', e); }

      // Ensure overlays are attached after base image opens
      try{ syncNucleiSegOverlay(); }catch(e){ console.warn('[overlay] sync nuclei_seg failed', e); }
      try{ syncTissueMaskOverlay(); }catch(e){ console.warn('[overlay] sync tissue_mask failed', e); }
      try{ syncTissueSegFullOverlay(); }catch(e){ console.warn('[overlay] sync tissue_seg_full failed', e); }
    });

    // keep scale/zoom updated during interactions
    viewer.addHandler('animation', ()=>{ updateScaleBar(); updateZoomLabel(); updatePyramidActive(); });
    viewer.addHandler('zoom', ()=>{ updatePyramidActive(); });
    viewer.addHandler('pan', ()=>{ updatePyramidActive(); });

    // helper to build and update pyramid controls
    // Return pyramid levels as array of objects { level: idx, width, height, downsample }    // User override timer prevents UI from snapping back while user is interacting
    let pyramidUserOverrideUntil = 0;    function computePyramidLevels(){
      try{
        // Prefer server-provided slide pyramid if available
        if(window.slidePyramid && window.slidePyramid.length){
          return window.slidePyramid.map(p=>({ level: p.level, width: p.width, height: p.height, downsample: p.downsample }));
        }
        if(!viewer || !viewer.world) return [{level:0, width:1, height:1, downsample:1}];
        const contentSize = viewer.world.getItemAt(0).getContentSize();
        const maxDim = Math.max(contentSize.x, contentSize.y) || 1;
        const levels = [];
        let idx = 0;
        for(let v=1; v<=maxDim; v*=2){ levels.push({level: idx++, width: Math.max(1, Math.floor(maxDim / v)), height: Math.max(1, Math.floor(maxDim / v)), downsample: v}); if(v>=4096) break; }
        return levels;
      }catch(e){ return [{level:0, width:1, height:1, downsample:1}]; }
    }

    function updatePyramidControls(){
      try{
        const container = viewer && viewer.container ? viewer.container : document.getElementById('viewer');
        if(!container) return;
        let pc = container.querySelector('#pyramid-controls');
        if(pc){ pc.parentElement.removeChild(pc); }
        pc = document.createElement('div'); pc.id = 'pyramid-controls'; pc.className = 'pyramid-slider';

        const levels = computePyramidLevels();
        if(!levels || levels.length===0) return;
        const label = document.createElement('div'); label.className = 'pyr-label'; label.textContent = `L${levels[levels.length-1].level} ×${levels[levels.length-1].downsample}`;
        const sliderWrap = document.createElement('div'); sliderWrap.className = 'pyr-slider';
        const slider = document.createElement('input'); slider.type = 'range'; slider.min = 0; slider.max = Math.max(0, levels.length - 1); slider.step = 1; slider.value = levels.length - 1;
        slider.setAttribute('aria-label', 'Pyramid level');
        slider.addEventListener('input', (e)=>{
          const idx = parseInt(e.target.value);
          const lvl = levels[idx];
          label.textContent = `L${lvl.level} ×${lvl.downsample}`;
          // User interacting - prevent UI auto-snap during interaction
          pyramidUserOverrideUntil = Date.now() + 300;
        });
        slider.addEventListener('pointerdown', ()=>{ pyramidUserOverrideUntil = Date.now() + 500; });
        slider.addEventListener('pointerup', (e)=>{ pyramidUserOverrideUntil = Date.now() + 1000; });
        slider.addEventListener('change', (e)=>{
          const idx = parseInt(e.target.value);
          const lvl = levels[idx];
          pyramidUserOverrideUntil = Date.now() + 1000; // keep override while animating
          zoomToDownsample(lvl.downsample, idx);
        });

        sliderWrap.appendChild(slider);
        pc.appendChild(label);
        pc.appendChild(sliderWrap);
        container.appendChild(pc);

        // store levels for later use
        pc._levels = levels;
        updatePyramidActive();
      }catch(e){ console.warn('[pyramid] updatePyramidControls error', e); }
    }

    function getCurrentImagePixelPerScreen(){
      if(!viewer || !viewer.world) return 1;
      const contentSize = viewer.world.getItemAt(0).getContentSize(); const imageWidth = contentSize.x || 1;
      const bounds = viewer.viewport.getBounds(true); const vpWidth = bounds.width; const containerPx = viewer.container.clientWidth || 1;
      return (imageWidth * vpWidth) / containerPx; // image pixels per screen pixel
    }

    function updatePyramidActive(){
      try{
        const pc = viewer && viewer.container ? viewer.container.querySelector('#pyramid-controls') : document.getElementById('pyramid-controls');
        if(!pc || !pc._levels) return;
        const levels = pc._levels;
        const current = getCurrentImagePixelPerScreen();
        // find nearest index
        let bestIdx = 0; let bestDiff = Infinity;
        levels.forEach((d, idx)=>{ const diff = Math.abs(d.downsample - current); if(diff < bestDiff){ bestDiff = diff; bestIdx = idx; } });
        const slider = pc.querySelector('input[type=range]'); const label = pc.querySelector('.pyr-label');
        // if user is interacting, don't override the slider position to avoid snapping back
        if(Date.now() < (pyramidUserOverrideUntil || 0)) return;
        if(slider){ slider.value = bestIdx; }
        if(label){ label.textContent = `L${levels[bestIdx].level} ×${levels[bestIdx].downsample}`; }
      }catch(e){ }
    }

    function zoomToDownsample(down, idx){
      try{
        if(!viewer || !viewer.world) return;
        const contentSize = viewer.world.getItemAt(0).getContentSize(); const imageWidth = contentSize.x || 1;
        const containerPx = viewer.container.clientWidth || 1;
        const desiredVpWidth = (down * containerPx) / imageWidth;
        const bounds = viewer.viewport.getBounds(true);
        const center = bounds.getCenter();
        const newBounds = new OpenSeadragon.Rect(center.x - desiredVpWidth/2, center.y - (bounds.height/2), desiredVpWidth, bounds.height);
        // set slider to requested index immediately so UI reflects user action
        try{ const pc = viewer.container.querySelector('#pyramid-controls'); if(pc){ const slider = pc.querySelector('input[type=range]'); const label = pc.querySelector('.pyr-label'); if(slider && typeof idx !== 'undefined'){ slider.value = idx; } if(label){ label.textContent = `×${down}`; } } }catch(e){}
        pyramidUserOverrideUntil = Date.now() + 1000;
        viewer.viewport.fitBounds(newBounds, true);
      }catch(e){ console.warn('[pyramid] zoomToDownsample error', e); }
    }

    // keep scale/zoom updated during interactions
    viewer.addHandler('animation', ()=>{ updateScaleBar(); updateZoomLabel(); });
    viewer.addHandler('tile-loaded', ()=>{ diag('[viewer] tile-loaded'); console.log('[viewer] tile-loaded'); });

    window.addEventListener('resize', ()=>{ updateScaleBar(); updateZoomLabel(); });
  }catch(err){ diag('[viewer] init error: ' + err); console.error('[viewer] init error', err); }
}

function getImagePixelsPerScreen(){
  if(!viewer || !viewer.viewport || !viewer.container) return 1;
  try{
    // Map [0,0] and [1,0] viewer-element pixels to image coordinates and return delta x
    const p0 = viewer.viewport.viewerElementToViewportCoordinates(new OpenSeadragon.Point(0,0));
    const i0 = viewer.viewport.viewportToImageCoordinates(p0);
    const p1 = viewer.viewport.viewerElementToViewportCoordinates(new OpenSeadragon.Point(1,0));
    const i1 = viewer.viewport.viewportToImageCoordinates(p1);
    const delta = Math.abs((i1 && i1.x || 0) - (i0 && i0.x || 0));
    return delta || 1;
  }catch(e){ return 1; }
}

function updateScaleBar(){
  const bar = document.getElementById('bar');
  const label = document.getElementById('barlabel');
  // Guard for missing DOM elements (avoid errors when switching slides/layout changes)
  if(!bar || !label) return;
  if(!viewer || !viewer.world){ bar.style.display='none'; return; }
  if(!mpp){ bar.style.display='none'; label.textContent=''; return; }
  bar.style.display='block';
  // candidate lengths in microns
  const candidates = [5,10,20,50,100,200,500,1000,2000,5000];

  const imagePixelPerScreenPx = getImagePixelsPerScreen();

  // pick candidate where displayed width is between ~60 and 180 px
  let chosen = candidates[candidates.length - 1];
  for(let c of candidates){
    const px = (c / mpp) / imagePixelPerScreenPx;
    if(px >= 60 && px <= 180){ chosen = c; break; }
    if(px > 180){ chosen = c; break; }
  }
  let pxWidth = Math.round((chosen / mpp) / imagePixelPerScreenPx);
  if(pxWidth < 30) pxWidth = 30;
  bar.style.width = pxWidth + 'px';

  // Nice label: use mm if >=1000 µm
  let labelText = '';
  if(chosen >= 1000){
    const mm = chosen / 1000.0;
    labelText = mm % 1 === 0 ? `${mm.toFixed(0)} mm` : `${mm.toFixed(2)} mm`;
  } else {
    labelText = `${chosen} µm`;
  }
  label.textContent = labelText;
}

function nucleiSegOverlayTileSource(){
  if(!slideName) return null;
  // Prefer DZI overlay if available (fast: tiled pyramid, like SVS)
  if(nucleiSegDziAvailable){
    return `/overlay_dzi/nuclei_seg/${encodeURIComponent(slideName)}.dzi`;
  }
  // Fallback: precomputed downsampled preview PNG
  return { type: 'image', url: `/overlay/nuclei_seg/${encodeURIComponent(slideName)}.png?down=${NUCLEI_SEG_DOWNSAMPLE}` };
}

function removeNucleiSegOverlay(){
  try{
    if(!viewer || !viewer.world) return;
    // remove any tracked overlay items (guards against duplicates)
    if(nucleiSegItems && nucleiSegItems.length){
      nucleiSegItems.forEach(it=>{ try{ viewer.world.removeItem(it); }catch(e){} });
    }
    if(nucleiSegItem){
      try{ viewer.world.removeItem(nucleiSegItem); }catch(e){}
    }
    nucleiSegItems = [];
    nucleiSegItem = null;
    nucleiSegLoading = false;
    try{ viewer.forceRedraw(); }catch(e){}
  }catch(e){ nucleiSegItem = null; }
}

function syncNucleiSegOverlay(){
  if(!viewer) return;
  if(!nucleiSegEnabled || !nucleiSegAvailable){
    removeNucleiSegOverlay();
    return;
  }
  if(nucleiSegLoading){
    // Prevent double-add while request in flight (open event + toggle can race)
    return;
  }
  // if already present, just update opacity
  if(nucleiSegItem){
    try{ nucleiSegItem.setOpacity(nucleiSegOpacity); }catch(e){}
    return;
  }
  const ts = nucleiSegOverlayTileSource();
  if(!ts) return;
  diag('[overlay] add nuclei_seg ' + (typeof ts === 'string' ? ts : ts.url));
  nucleiSegLoading = true;
  viewer.addTiledImage({
    tileSource: ts,
    x: 0,
    y: 0,
    width: 1,
    opacity: nucleiSegOpacity,
    success: function(ev){
      nucleiSegLoading = false;
      const item = ev.item;
      // If user turned it off while loading, remove immediately.
      if(!nucleiSegEnabled || !nucleiSegAvailable){
        try{ viewer.world.removeItem(item); }catch(e){}
        try{ viewer.forceRedraw(); }catch(e){}
        diag('[overlay] nuclei_seg loaded but disabled -> removed');
        return;
      }
      nucleiSegItem = item;
      nucleiSegItems.push(item);
      try{ nucleiSegItem.setOpacity(nucleiSegOpacity); }catch(e){}
      diag('[overlay] nuclei_seg added');
    },
    error: function(){
      // If server returns 404/500, mark unavailable so UI updates.
      nucleiSegLoading = false;
      nucleiSegAvailable = false;
      updateOverlayAvailabilityUI();
      removeNucleiSegOverlay();
      diag('[overlay] nuclei_seg failed to load');
    }
  });
}

function tissueMaskOverlayTileSource(){
  if(!slideName) return null;
  const tag = tissueMaskColorTag ? encodeURIComponent(String(tissueMaskColorTag)) : '';
  const vq = tag ? `&v=${tag}` : '';
  return { type: 'image', url: `/overlay/tissue_mask/${encodeURIComponent(slideName)}.png?down=${TISSUE_MASK_DOWNSAMPLE}${vq}` };
}

function tissueSegFullOverlayTileSource(){
  if(!slideName) return null;
  return { type: 'image', url: `/overlay/tissue_seg_full/${encodeURIComponent(slideName)}.png` };
}

function removeTissueSegFullOverlay(){
  try{
    if(!viewer || !viewer.world) return;
    if(tissueSegFullItems && tissueSegFullItems.length){
      tissueSegFullItems.forEach(it=>{ try{ viewer.world.removeItem(it); }catch(e){} });
    }
    if(tissueSegFullItem){
      try{ viewer.world.removeItem(tissueSegFullItem); }catch(e){}
    }
    tissueSegFullItems = [];
    tissueSegFullItem = null;
    tissueSegFullLoading = false;
    try{ viewer.forceRedraw(); }catch(e){}
  }catch(e){ tissueSegFullItem = null; }
}

function syncTissueSegFullOverlay(){
  if(!viewer) return;
  if(!tissueSegFullEnabled || !tissueSegFullAvailable){
    removeTissueSegFullOverlay();
    return;
  }
  if(tissueSegFullLoading) return;
  if(tissueSegFullItem){
    try{ tissueSegFullItem.setOpacity(tissueSegFullOpacity); }catch(e){}
    return;
  }
  const ts = tissueSegFullOverlayTileSource();
  if(!ts) return;
  diag('[overlay] add tissue_seg_full ' + ts.url);
  tissueSegFullLoading = true;
  viewer.addTiledImage({
    tileSource: ts,
    x: 0,
    y: 0,
    width: 1,
    opacity: tissueSegFullOpacity,
    success: function(ev){
      tissueSegFullLoading = false;
      const item = ev.item;
      if(!tissueSegFullEnabled || !tissueSegFullAvailable){
        try{ viewer.world.removeItem(item); }catch(e){}
        try{ viewer.forceRedraw(); }catch(e){}
        diag('[overlay] tissue_seg_full loaded but disabled -> removed');
        return;
      }
      tissueSegFullItem = item;
      tissueSegFullItems.push(item);
      try{ tissueSegFullItem.setOpacity(tissueSegFullOpacity); }catch(e){}
      diag('[overlay] tissue_seg_full added');
    },
    error: function(){
      tissueSegFullLoading = false;
      tissueSegFullAvailable = false;
      updateOverlayAvailabilityUI();
      removeTissueSegFullOverlay();
      diag('[overlay] tissue_seg_full failed to load');
    }
  });
}

function removeTissueMaskOverlay(){
  try{
    if(!viewer || !viewer.world) return;
    if(tissueMaskItems && tissueMaskItems.length){
      tissueMaskItems.forEach(it=>{ try{ viewer.world.removeItem(it); }catch(e){} });
    }
    if(tissueMaskItem){
      try{ viewer.world.removeItem(tissueMaskItem); }catch(e){}
    }
    tissueMaskItems = [];
    tissueMaskItem = null;
    tissueMaskLoading = false;
    try{ viewer.forceRedraw(); }catch(e){}
  }catch(e){ tissueMaskItem = null; }
}

function syncTissueMaskOverlay(){
  if(!viewer) return;
  if(!tissueMaskEnabled || !tissueMaskAvailable){
    removeTissueMaskOverlay();
    return;
  }
  if(tissueMaskLoading) return;
  if(tissueMaskItem){
    try{ tissueMaskItem.setOpacity(tissueMaskOpacity); }catch(e){}
    return;
  }
  const ts = tissueMaskOverlayTileSource();
  if(!ts) return;
  diag('[overlay] add tissue_mask ' + ts.url);
  tissueMaskLoading = true;
  viewer.addTiledImage({
    tileSource: ts,
    x: 0,
    y: 0,
    width: 1,
    opacity: tissueMaskOpacity,
    success: function(ev){
      tissueMaskLoading = false;
      const item = ev.item;
      if(!tissueMaskEnabled || !tissueMaskAvailable){
        try{ viewer.world.removeItem(item); }catch(e){}
        try{ viewer.forceRedraw(); }catch(e){}
        diag('[overlay] tissue_mask loaded but disabled -> removed');
        return;
      }
      tissueMaskItem = item;
      tissueMaskItems.push(item);
      try{ tissueMaskItem.setOpacity(tissueMaskOpacity); }catch(e){}
      diag('[overlay] tissue_mask added');
    },
    error: function(){
      tissueMaskLoading = false;
      tissueMaskAvailable = false;
      updateOverlayAvailabilityUI();
      removeTissueMaskOverlay();
      diag('[overlay] tissue_mask failed to load');
    }
  });
}

function initOverlayControls(){
  const toggle = document.getElementById('toggle-nuclei-seg');
  const slider = document.getElementById('nuclei-seg-opacity');
  const val = document.getElementById('nuclei-seg-opacity-val');
  if(!toggle || !slider || !val) return;

  // restore persisted settings
  try{
    const savedOn = localStorage.getItem('nucleiSegEnabled');
    const savedOp = localStorage.getItem('nucleiSegOpacity');
    if(savedOn === '1') nucleiSegEnabled = true;
    if(savedOp !== null){
      const f = parseFloat(savedOp);
      if(!Number.isNaN(f)) nucleiSegOpacity = Math.max(0, Math.min(1, f));
    }
  }catch(e){}

  toggle.checked = nucleiSegEnabled;
  slider.value = nucleiSegOpacity.toFixed(2);
  val.textContent = nucleiSegOpacity.toFixed(2);

  toggle.addEventListener('change', ()=>{
    nucleiSegEnabled = !!toggle.checked;
    try{ localStorage.setItem('nucleiSegEnabled', nucleiSegEnabled ? '1':'0'); }catch(e){}
    syncNucleiSegOverlay();
  });
  slider.addEventListener('input', ()=>{
    nucleiSegOpacity = Math.max(0, Math.min(1, parseFloat(slider.value)));
    val.textContent = nucleiSegOpacity.toFixed(2);
    try{ localStorage.setItem('nucleiSegOpacity', String(nucleiSegOpacity)); }catch(e){}
    if(nucleiSegItem){ try{ nucleiSegItem.setOpacity(nucleiSegOpacity); }catch(e){} }
  });

  updateOverlayAvailabilityUI();

  // tissue mask controls
  const tToggle = document.getElementById('toggle-tissue-mask');
  const tSlider = document.getElementById('tissue-mask-opacity');
  const tVal = document.getElementById('tissue-mask-opacity-val');
  if(tToggle && tSlider && tVal){
    try{
      const savedOn = localStorage.getItem('tissueMaskEnabled');
      const savedOp = localStorage.getItem('tissueMaskOpacity');
      if(savedOn === '1') tissueMaskEnabled = true;
      if(savedOp !== null){
        const f = parseFloat(savedOp);
        if(!Number.isNaN(f)) tissueMaskOpacity = Math.max(0, Math.min(1, f));
      }
    }catch(e){}

    tToggle.checked = tissueMaskEnabled;
    tSlider.value = tissueMaskOpacity.toFixed(2);
    tVal.textContent = tissueMaskOpacity.toFixed(2);

    tToggle.addEventListener('change', ()=>{
      tissueMaskEnabled = !!tToggle.checked;
      try{ localStorage.setItem('tissueMaskEnabled', tissueMaskEnabled ? '1':'0'); }catch(e){}
      syncTissueMaskOverlay();
    });
    tSlider.addEventListener('input', ()=>{
      tissueMaskOpacity = Math.max(0, Math.min(1, parseFloat(tSlider.value)));
      tVal.textContent = tissueMaskOpacity.toFixed(2);
      try{ localStorage.setItem('tissueMaskOpacity', String(tissueMaskOpacity)); }catch(e){}
      if(tissueMaskItem){ try{ tissueMaskItem.setOpacity(tissueMaskOpacity); }catch(e){} }
    });
  }

  // tissue_seg/full_images controls
  const sToggle = document.getElementById('toggle-tissue-seg-full');
  const sSlider = document.getElementById('tissue-seg-full-opacity');
  const sVal = document.getElementById('tissue-seg-full-opacity-val');
  if(sToggle && sSlider && sVal){
    try{
      const savedOn = localStorage.getItem('tissueSegFullEnabled');
      const savedOp = localStorage.getItem('tissueSegFullOpacity');
      if(savedOn === '1') tissueSegFullEnabled = true;
      if(savedOp !== null){
        const f = parseFloat(savedOp);
        if(!Number.isNaN(f)) tissueSegFullOpacity = Math.max(0, Math.min(1, f));
      }
    }catch(e){}

    sToggle.checked = tissueSegFullEnabled;
    sSlider.value = tissueSegFullOpacity.toFixed(2);
    sVal.textContent = tissueSegFullOpacity.toFixed(2);

    sToggle.addEventListener('change', ()=>{
      tissueSegFullEnabled = !!sToggle.checked;
      try{ localStorage.setItem('tissueSegFullEnabled', tissueSegFullEnabled ? '1':'0'); }catch(e){}
      syncTissueSegFullOverlay();
    });
    sSlider.addEventListener('input', ()=>{
      tissueSegFullOpacity = Math.max(0, Math.min(1, parseFloat(sSlider.value)));
      sVal.textContent = tissueSegFullOpacity.toFixed(2);
      try{ localStorage.setItem('tissueSegFullOpacity', String(tissueSegFullOpacity)); }catch(e){}
      if(tissueSegFullItem){ try{ tissueSegFullItem.setOpacity(tissueSegFullOpacity); }catch(e){} }
    });
  }
}

function updateOverlayAvailabilityUI(){
  const toggle = document.getElementById('toggle-nuclei-seg');
  const hint = document.getElementById('nuclei-seg-hint');
  if(toggle){
    toggle.disabled = !nucleiSegAvailable;
    if(!nucleiSegAvailable){
      toggle.checked = false;
      nucleiSegEnabled = false;
      try{ localStorage.setItem('nucleiSegEnabled','0'); }catch(e){}
      removeNucleiSegOverlay();
    }
  }
  if(hint){
    hint.style.display = nucleiSegAvailable ? 'none' : 'block';
  }

  const tToggle = document.getElementById('toggle-tissue-mask');
  const tHint = document.getElementById('tissue-mask-hint');
  if(tToggle){
    tToggle.disabled = !tissueMaskAvailable;
    if(!tissueMaskAvailable){
      tToggle.checked = false;
      tissueMaskEnabled = false;
      try{ localStorage.setItem('tissueMaskEnabled','0'); }catch(e){}
      removeTissueMaskOverlay();
    }
  }
  if(tHint){
    tHint.style.display = tissueMaskAvailable ? 'none' : 'block';
  }

  const sToggle = document.getElementById('toggle-tissue-seg-full');
  const sHint = document.getElementById('tissue-seg-full-hint');
  if(sToggle){
    sToggle.disabled = !tissueSegFullAvailable;
    if(!tissueSegFullAvailable){
      sToggle.checked = false;
      tissueSegFullEnabled = false;
      try{ localStorage.setItem('tissueSegFullEnabled','0'); }catch(e){}
      removeTissueSegFullOverlay();
    }
  }
  if(sHint){
    sHint.style.display = tissueSegFullAvailable ? 'none' : 'block';
  }
}

slidesSelect.addEventListener('change', ()=> setSlide(slidesSelect.value));
reloadBtn.addEventListener('click', ()=> fetchSlides());

// Sidebar control and layout update (initialized once)
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
function setSidebarCollapsed(collapsed){
  if(collapsed) sidebar.classList.add('collapsed'); else sidebar.classList.remove('collapsed');
  localStorage.setItem('sidebarCollapsed', collapsed?'1':'0');
  updateViewerLayout();
}
if(sidebarToggle){
  sidebarToggle.addEventListener('click', ()=>{ setSidebarCollapsed(!sidebar.classList.contains('collapsed')); });
  // restore saved state only when toggle exists (otherwise user can't reopen it)
  try{ if(localStorage.getItem('sidebarCollapsed')==='1') sidebar.classList.add('collapsed') }catch(e){}
} else {
  // force expanded
  try{ sidebar.classList.remove('collapsed'); localStorage.setItem('sidebarCollapsed','0'); }catch(e){}
}

// Rightbar control (behavior mirrors left sidebar)
const rightbar = document.getElementById('rightbar');
const rightbarToggle = document.getElementById('rightbar-toggle');
function setRightbarCollapsed(collapsed){
  if(!rightbar) return;
  if(collapsed) rightbar.classList.add('collapsed'); else rightbar.classList.remove('collapsed');
  if(rightbarToggle) rightbarToggle.classList.toggle('collapsed', collapsed);
  localStorage.setItem('rightbarCollapsed', collapsed?'1':'0');
  updateViewerLayout();
}
if(rightbarToggle) rightbarToggle.addEventListener('click', ()=> setRightbarCollapsed(!rightbar.classList.contains('collapsed')));
// restore saved state
try{ if(localStorage.getItem('rightbarCollapsed')==='1') rightbar.classList.add('collapsed') }catch(e){}

function updateViewerLayout(){
  // adjust viewer left position to accommodate sidebar width when expanded
  const viewerEl = document.getElementById('viewer');
  if(!viewerEl) return;
  const leftCollapsed = sidebar.classList.contains('collapsed');
  const rightCollapsed = rightbar && rightbar.classList.contains('collapsed');
  if(leftCollapsed){ viewerEl.style.left = '20px'; }
  else { viewerEl.style.left = (sidebar.offsetWidth + 20) + 'px'; }
  if(rightbar){
    if(rightCollapsed){ viewerEl.style.right = '20px'; }
    else { viewerEl.style.right = (rightbar.offsetWidth + 20) + 'px'; }
  } else {
    viewerEl.style.right = '20px';
  }
  // if OpenSeadragon viewer exists, notify it to redraw
  try{ if(window.viewer && window.viewer instanceof OpenSeadragon.Viewer){ window.viewer.forceRedraw(); } }catch(e){}
}
// keep layout updated on resize
window.addEventListener('resize', updateViewerLayout);
// call once to ensure correct layout
updateViewerLayout();

// Upload handling (auto-upload on selection)
if(fileInput){
  fileInput.addEventListener('change', ()=>{
    try{
      const statusEl = document.getElementById('fileinput-status');
      const f = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
      if(statusEl){
        statusEl.textContent = f ? f.name : 'No file selected';
      }
    }catch(e){}
    uploadSlide();
  });
}

async function uploadSlide(){
  const f = fileInput.files[0];
  if(!f){ alert('Select a file to upload'); return; }
  const fd = new FormData(); fd.append('slide', f);
  if(fileInput) fileInput.disabled = true;
  try{
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const j = await r.json();
    if(j.ok){ alert('Uploaded: ' + j.filename); fetchSlides(); }
    else alert('Upload failed: ' + (j.error||'unknown'));
  }catch(e){ alert('Upload error: '+e); }
  if(fileInput){
    fileInput.disabled = false;
    try{ fileInput.value=''; }catch(e){}
    try{
      const statusEl = document.getElementById('fileinput-status');
      if(statusEl) statusEl.textContent = 'No file selected';
    }catch(e){}
  }
}

// Management list
function updateManageListActive(activeName){
  try{
    const items = document.querySelectorAll('#manage-list .manage-item');
    items.forEach(el => {
      if(el.dataset.slide === activeName) el.classList.add('manage-item-active');
      else el.classList.remove('manage-item-active');
    });
  }catch(e){}
}
function renderManage(list){
  const box = document.getElementById('manage-list'); box.innerHTML = '';
  list.forEach(name => {
    const item = document.createElement('div'); item.className='manage-item'; item.dataset.slide = name;
    if(name === slideName) item.classList.add('manage-item-active');
    const img = document.createElement('img'); img.className='manage-thumb'; img.src=`/thumbnail/${encodeURIComponent(name)}.jpg`;
    const span = document.createElement('div'); span.className='manage-name'; span.textContent = name;
    const actions = document.createElement('div'); actions.className='manage-actions';
    const btnOpen = document.createElement('button');
    btnOpen.textContent='Open';
    btnOpen.className = 'btn-open';
    btnOpen.addEventListener('click', ()=> { slidesSelect.value = name; setSlide(name); updateManageListActive(name); });

    // Delete is intentionally disabled in this build (prevent accidental data loss).
    // Keep it clickable for a consistent layout, but show an explicit message.
    const btnDelete = document.createElement('button');
    btnDelete.textContent = 'Delete';
    btnDelete.className = 'btn-delete-disabled';
    btnDelete.setAttribute('aria-label', 'Delete (disabled)');
    btnDelete.title = 'Delete (disabled)';
    btnDelete.addEventListener('click', async (e)=>{
      try{ e.preventDefault(); e.stopPropagation(); }catch(err){}
      await showInfo('Delete is disabled in this build.', 'Disabled');
    });
    // Swap button order: Delete (left) then Open (right)
    actions.appendChild(btnDelete); actions.appendChild(btnOpen);
    item.appendChild(img); item.appendChild(span); item.appendChild(actions);
    box.appendChild(item);
  });
}

// --- Patch: updateZoomLabel uses correct display magnification ---
function updateZoomLabel(){
  const optEl = document.getElementById('zoom-opt-val');
  const effEl = document.getElementById('zoom-eff-val');
  if(!optEl || !effEl) return;
  if(!viewer || !viewer.world){ optEl.textContent='—'; effEl.textContent='—'; return; }

  // Use actual pyramid levels if available
  let downsample = 1;
  if(window.slidePyramid && window.slidePyramid.length) {
    // Find the closest level to current view
    const imagePixelPerScreenPx = getImagePixelsPerScreen();
    let bestIdx = 0, bestDiff = Infinity;
    window.slidePyramid.forEach((lvl, idx) => {
      const diff = Math.abs(lvl.downsample - imagePixelPerScreenPx);
      if(diff < bestDiff) { bestDiff = diff; bestIdx = idx; }
    });
    downsample = window.slidePyramid[bestIdx].downsample || 1;
  } else {
    // fallback: estimate from current zoom
    const contentSize = viewer.world.getItemAt(0).getContentSize();
    const imageWidth = contentSize.x || 1;
    const bounds = viewer.viewport.getBounds(true);
    const vpWidth = bounds.width;
    const containerPx = viewer.container.clientWidth || 1;
    const imagePixelPerScreenPx = (imageWidth * vpWidth) / containerPx;
    downsample = imagePixelPerScreenPx;
  }

  if(objective) {
    optEl.textContent = `×${objective}`;
    // Display magnification = objective / downsample
    const displayMag = objective / downsample;
    effEl.textContent = `×${displayMag.toFixed(2)}`;
    document.getElementById('zoombox').title = `Optical: ×${objective}; Display: ×${displayMag.toFixed(2)} (objective / downsample=${downsample})`;
    return;
  }

  if(mpp) {
    const approxObj = Math.max(1, Math.round(10.0 / mpp));
    optEl.textContent = `≈×${approxObj}`;
    const displayMag = approxObj / downsample;
    effEl.textContent = `×${displayMag.toFixed(2)}`;
    document.getElementById('zoombox').title = `Estimated optical: ≈×${approxObj} (based on mpp=${mpp} µm/px); Display: ×${displayMag.toFixed(2)}`;
    return;
  }

  // fallback: show percentage in effective and dash for optical
  optEl.textContent = '—';
  effEl.textContent = '—';
}
// --- End patch ---

function showModal(opts){
  return new Promise((resolve) => {
    const modal = document.getElementById('confirm-modal');
    const titleEl = document.getElementById('confirm-title');
    const msgEl = document.getElementById('confirm-message');
    const btnCancel = document.getElementById('confirm-cancel');
    const btnOk = document.getElementById('confirm-ok');
    if(!modal || !msgEl || !btnCancel || !btnOk){
      // Fallback (should not happen): avoid native confirm() to prevent OS-localized buttons.
      resolve(!!(opts && opts.fallbackResult));
      return;
    }
    const o = opts || {};
    if(titleEl) titleEl.textContent = o.title || 'Confirm';
    msgEl.textContent = o.message || '';
    btnOk.textContent = o.okLabel || 'OK';
    btnCancel.textContent = o.cancelLabel || 'Cancel';
    const showCancel = (o.showCancel !== false);
    btnCancel.style.display = showCancel ? '' : 'none';

    const cleanup = () => {
      try{ modal.classList.add('hidden'); }catch(e){}
      btnCancel.onclick = null;
      btnOk.onclick = null;
      modal.onkeydown = null;
      try{ modal.removeEventListener('click', onBackdropClick); }catch(e){}
    };

    const finish = (val) => { cleanup(); resolve(val); };

    const onBackdropClick = (e) => {
      try{
        const t = e.target;
        if(t && t.getAttribute && t.getAttribute('data-modal-close') === '1'){
          finish(false);
        }
      }catch(err){}
    };

    btnCancel.onclick = ()=> finish(false);
    btnOk.onclick = ()=> finish(true);
    modal.onkeydown = (e)=>{ if(e && e.key === 'Escape') finish(false); };
    modal.addEventListener('click', onBackdropClick);

    modal.classList.remove('hidden');
    // Focus the safer option by default
    try{ (showCancel ? btnCancel : btnOk).focus(); }catch(e){}
  });
}

function showInfo(message, title){
  return showModal({ title: title || 'Info', message: message || '', okLabel: 'OK', showCancel: false, fallbackResult: true });
}

function confirmDelete(name){
  // Use custom modal so buttons are always English.
  return showModal({
    title: 'Confirm delete',
    message: `Delete \"${name}\"? This cannot be undone.`,
    okLabel: 'Delete',
    cancelLabel: 'Cancel',
    showCancel: true,
    fallbackResult: false,
  });
}

async function deleteSlide(name){
  const ok = await confirmDelete(name);
  if(!ok) return;
  try{
    const r = await fetch('/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ slide: name }) });
    const j = await r.json();
    if(j.ok){ alert('Deleted'); fetchSlides(); }
    else alert('Delete failed: '+(j.error||'unknown'));
  }catch(e){ alert('Delete error: '+e); }
}

fetchSlides();

// Update zoom label periodically (useful for smooth animated zooms)
setInterval(()=>{ try{ updateZoomLabel(); }catch(e){} }, 500);

/* ------------------ Annotation tools (disabled in this build) ------------------ */

let annotations = []; // stored in image coordinates
let annIdCounter = 1;
let currentTool = 'pan';
let drawing = false;
let currentRectStart = null; // image coords
let currentPolygonPoints = []; // image coords while drawing
let svgOverlay = null;
let _lastMouseEvent = null;
// annotation click/shape debug toggle (persisted)
let annotationDebug = (localStorage.getItem('annoDebug')==='1');
window.toggleAnnotationDebug = function(){ annotationDebug = !annotationDebug; localStorage.setItem('annoDebug', annotationDebug ? '1':'0'); diag('[annotations] debug ' + (annotationDebug ? 'ON':'OFF')); if(!annotationDebug && window._annotationDebugGroup){ while(window._annotationDebugGroup.firstChild) window._annotationDebugGroup.removeChild(window._annotationDebugGroup.firstChild); } };

function clearDebugMarkers(){ try{ if(window._annotationDebugGroup){ while(window._annotationDebugGroup.firstChild) window._annotationDebugGroup.removeChild(window._annotationDebugGroup.firstChild); } }catch(e){} }
function drawDebugCross(x,y,color,size=8,thickness=1){ try{ if(!window._annotationDebugGroup) return; const g = window._annotationDebugGroup; const l1 = document.createElementNS('http://www.w3.org/2000/svg','line'); l1.setAttribute('x1',x-size); l1.setAttribute('y1',y); l1.setAttribute('x2',x+size); l1.setAttribute('y2',y); l1.setAttribute('stroke',color); l1.setAttribute('stroke-width',thickness); l1.setAttribute('pointer-events','none'); g.appendChild(l1); const l2 = document.createElementNS('http://www.w3.org/2000/svg','line'); l2.setAttribute('x1',x); l2.setAttribute('y1',y-size); l2.setAttribute('x2',x); l2.setAttribute('y2',y+size); l2.setAttribute('stroke',color); l2.setAttribute('stroke-width',thickness); l2.setAttribute('pointer-events','none'); g.appendChild(l2); }catch(e){}}

function initAnnotationsOverlay(){
  // create overlay attached to OpenSeadragon viewer container (on-demand)
  const container = (window.viewer && window.viewer.container) ? window.viewer.container : document.getElementById('viewer');
  if(!container){ console.warn('[annotations] initAnnotationsOverlay: no container found'); return; }

  // If an existing overlay is attached to a different container, remove it so we recreate correctly
  if(svgOverlay && svgOverlay.parentElement && svgOverlay.parentElement !== container){
    diag('[annotations] existing overlay attached to old container - removing and recreating');
    try{ svgOverlay.parentElement.removeChild(svgOverlay); }catch(e){}
    svgOverlay = null;
  }

  if(svgOverlay) return; // exist and attached to correct container

  svgOverlay = document.createElementNS('http://www.w3.org/2000/svg','svg');
  svgOverlay.setAttribute('class','annotation-overlay');
  // absolutely cover the viewer container
  svgOverlay.style.position='absolute'; svgOverlay.style.left='0'; svgOverlay.style.top='0'; svgOverlay.style.width='100%'; svgOverlay.style.height='100%';
  svgOverlay.style.pointerEvents='none'; // enabled when annotation mode active
  svgOverlay.style.zIndex = 999999; // very high to be above viewer content
  svgOverlay.style.cursor = 'crosshair';

  // ensure container is positioned so absolute children align
  const cs = window.getComputedStyle(container);
  if(cs.position === 'static') container.style.position = 'relative';

  container.appendChild(svgOverlay);

  // set explicit width/height and viewBox to match container for pixel coords
  function updateSVGSize(){
    try{
      const w = container.clientWidth || 1;
      const h = container.clientHeight || 1;
      svgOverlay.setAttribute('width', w);
      svgOverlay.setAttribute('height', h);
      svgOverlay.setAttribute('viewBox', `0 0 ${w} ${h}`);
    }catch(e){ }
  }
  updateSVGSize();

  // create debug group (used when annotationDebug is enabled)
  try{
    const dbgGroup = document.createElementNS('http://www.w3.org/2000/svg','g'); dbgGroup.setAttribute('id','annotation-debug'); svgOverlay.appendChild(dbgGroup); window._annotationDebugGroup = dbgGroup;
  }catch(e){ }

  // mouse handlers on overlay (attached once)
  svgOverlay.addEventListener('mousedown', onOverlayMouseDown);
  svgOverlay.addEventListener('mousemove', function(e){ _lastMouseEvent = e; onOverlayMouseMove(e); });
  svgOverlay.addEventListener('mouseup', onOverlayMouseUp);
  svgOverlay.addEventListener('click', onOverlayClick);

  // expose size updater
  window.updateOverlaySize = updateSVGSize;

  // re-render when viewer moves or opens (keep overlay in sync)
  if(window.viewer){
    window.viewer.addHandler('animation', renderAnnotations);
    window.viewer.addHandler('open', ()=>{
      renderAnnotations();
      // ensure overlay sits on top after viewer created
      try{ svgOverlay.style.zIndex = 999999; }catch(e){}
    });
  }

  // keep overlay size if container resizes
  window.addEventListener('resize', ()=>{ try{ updateSVGSize(); renderAnnotations(); }catch(e){} });
  diag('[annotations] overlay initialized');
}

function updateOverlaySize(){ try{ if(window.updateOverlaySize) window.updateOverlaySize(); }catch(e){} }


function initAnnotationControls(){
  console.log('[annotations] initAnnotationControls');
  const annTools = document.querySelectorAll('#rightbar .tool');
  const finishPolygonBtn = document.getElementById('finish-polygon');
  const saveBtn = document.getElementById('save-annotations');
  const loadInput = document.getElementById('load-annotations');
  const annListEl = document.getElementById('annotation-list');

  if(!annTools || !finishPolygonBtn || !saveBtn || !loadInput || !annListEl) return;

  annTools.forEach(b=> b.addEventListener('click', ()=> { diag('[annotations] tool clicked ' + b.dataset.tool); setTool(b.dataset.tool, annTools, finishPolygonBtn); }));
  finishPolygonBtn.addEventListener('click', ()=> finishPolygon());
  saveBtn.addEventListener('click', ()=> saveAnnotations());
  loadInput.addEventListener('change', (ev)=> loadAnnotationsFromFile(ev));

  // helper to render list with editable labels
  function renderAnnotationListLocal(){
    annListEl.innerHTML = '';
    annotations.forEach(a=>{
      const div = document.createElement('div'); div.className='annotation-item';
      const left = document.createElement('div'); left.style.cursor = 'pointer';
      const labelSpan = document.createElement('span'); labelSpan.className = 'annotation-label';
      labelSpan.textContent = a.label || `${a.type} ${a.id}`;
      labelSpan.title = 'Click to edit label';
      labelSpan.addEventListener('click', ()=>{
        const input = document.createElement('input'); input.type='text'; input.value = a.label || `${a.type} ${a.id}`;
        input.style.minWidth = '120px';
        input.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ input.blur(); } if(e.key==='Escape'){ input.value = a.label || `${a.type} ${a.id}`; input.blur(); } });
        input.addEventListener('blur', ()=>{ try{ const val = input.value.trim(); a.label = val || `${a.type} ${a.id}`; renderAnnotations(); if(window.renderAnnotationList) window.renderAnnotationList(); }catch(e){ console.warn('[annotations] label edit blur handler error', e); } });
        labelSpan.replaceWith(input); input.focus(); input.select();
      });
      left.appendChild(labelSpan);

      const right = document.createElement('div');
      const del = document.createElement('button'); del.textContent='Delete'; del.addEventListener('click', ()=> removeAnnotation(a.id));
      right.appendChild(del);
      div.appendChild(left); div.appendChild(right);
      annListEl.appendChild(div);
    });
  }
  // expose global helper
  window.renderAnnotationList = renderAnnotationListLocal;

  // default to pan tool
  setTool('pan', annTools, finishPolygonBtn);
}

function setTool(tool, annTools, finishPolygonBtn){
  console.log('[annotations] setTool', tool);
  currentTool = tool;
  if(annTools) annTools.forEach(b=> b.classList.toggle('active', b.dataset.tool===tool));
  initAnnotationsOverlay();
  const finishBtn = finishPolygonBtn || document.getElementById('finish-polygon');
  if(tool==='pan'){
    if(window.viewer) window.viewer.setMouseNavEnabled(true);
    if(svgOverlay){ svgOverlay.style.pointerEvents = 'none'; svgOverlay.style.cursor = 'default'; }
    if(finishBtn) finishBtn.style.display='none';
    diag('[annotations] setTool pan');
  } else {
    if(window.viewer) window.viewer.setMouseNavEnabled(false);
    initAnnotationsOverlay();
    if(svgOverlay){ svgOverlay.style.pointerEvents = 'auto'; svgOverlay.style.cursor = 'crosshair'; }
    if(finishBtn) finishBtn.style.display = tool==='polygon' ? 'inline-block' : 'none';
    diag('[annotations] setTool ' + tool);
  }
}

function onOverlayClick(ev){
  console.log('[annotations] overlay click, tool=', currentTool);
  const imgPt = viewerElementToImagePoint(ev.clientX, ev.clientY);
  if(!imgPt){ diag('[annotations] click ignored - viewer not ready'); return; }

  // Debug: show both the raw SVG click coordinates and the computed image->SVG point
  if(annotationDebug && svgOverlay && typeof svgOverlay.getScreenCTM === 'function'){
    try{
      clearDebugMarkers();
      const pt = svgOverlay.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
      const svgClick = pt.matrixTransform(svgOverlay.getScreenCTM().inverse());
      const imgSVG = imageToSVGPoint(imgPt);
      const dx = imgSVG.x - svgClick.x; const dy = imgSVG.y - svgClick.y;

      // Additional diagnostics: map image->viewer-element px and back from svgClick to viewer-element
      const viewportPt = viewer.viewport.imageToViewportCoordinates(new OpenSeadragon.Point(imgPt.x, imgPt.y));
      const viewerPx = viewer.viewport.viewportToViewerElementCoordinates(viewportPt);
      const rect = viewer.container.getBoundingClientRect();
      const clientFromViewerPx = { x: Math.round(rect.left + viewerPx.x), y: Math.round(rect.top + viewerPx.y) };

      // map svgClick back to screen coords to compare with client coords
      const ptSvgBack = svgOverlay.createSVGPoint(); ptSvgBack.x = svgClick.x; ptSvgBack.y = svgClick.y;
      const screenFromSvg = ptSvgBack.matrixTransform(svgOverlay.getScreenCTM());
      const viewerPxFromSvg = { x: Math.round(screenFromSvg.x - rect.left), y: Math.round(screenFromSvg.y - rect.top) };

      const z = viewer.viewport.getZoom();
      diag(`[annotations debug] zoom=${z.toFixed(4)} client(ev)=(${ev.clientX},${ev.clientY}) clientFromViewerPx=(${clientFromViewerPx.x},${clientFromViewerPx.y}) svgClick=(${svgClick.x.toFixed(1)},${svgClick.y.toFixed(1)}) imgSVG=(${imgSVG.x.toFixed(1)},${imgSVG.y.toFixed(1)}) dx=${dx.toFixed(2)} dy=${dy.toFixed(2)} viewerPx=(${viewerPx.x.toFixed(2)},${viewerPx.y.toFixed(2)}) viewerPxFromSvg=(${viewerPxFromSvg.x},${viewerPxFromSvg.y})`);

      drawDebugCross(svgClick.x, svgClick.y, 'red', 6, 1.5);
      drawDebugCross(imgSVG.x, imgSVG.y, 'blue', 6, 1.5);
      // also mark viewerPx computed position (converted to svg by viewerPx -> client -> svg)
      drawDebugCross(imgSVG.x, imgSVG.y, 'blue', 6, 1.5);
    }catch(e){ console.warn('[annotations] debug draw failed', e); }
  }

  if(currentTool==='point'){
    addAnnotation({type:'point', x: imgPt.x, y: imgPt.y});
  } else if(currentTool==='polygon'){
    currentPolygonPoints.push(imgPt);
    drawing = true;
    renderAnnotations();
  }
}

function onOverlayMouseDown(ev){
  if(currentTool!=='rect') return;
  const imgPt = viewerElementToImagePoint(ev.clientX, ev.clientY);
  if(!imgPt){ diag('[annotations] mousedown ignored - viewer not ready'); return; }
  drawing = true;
  currentRectStart = imgPt;
  ev.preventDefault(); ev.stopPropagation();
}
function onOverlayMouseMove(ev){
  if(!viewer) return;
  if(currentTool==='rect' && drawing && currentRectStart){
    // just re-render temp rect
    renderAnnotations();
  }
}
function onOverlayMouseUp(ev){
  if(currentTool!=='rect') return;
  if(drawing && currentRectStart){
    const imgPt = viewerElementToImagePoint(ev.clientX, ev.clientY);
    if(!imgPt){ diag('[annotations] mouseup ignored - viewer not ready'); return; }
    const x = Math.min(currentRectStart.x, imgPt.x);
    const y = Math.min(currentRectStart.y, imgPt.y);
    const w = Math.abs(currentRectStart.x - imgPt.x);
    const h = Math.abs(currentRectStart.y - imgPt.y);
    addAnnotation({type:'rect', x, y, w, h});
    drawing = false; currentRectStart = null; renderAnnotations();
    ev.preventDefault(); ev.stopPropagation();
  }
}

function finishPolygon(){
  if(currentPolygonPoints.length >= 3){
    addAnnotation({type:'polygon', points: currentPolygonPoints.slice()});
  }
  currentPolygonPoints = [];
  drawing = false;
  renderAnnotations();
}

function addAnnotation(a){
  a.id = 'a' + (annIdCounter++);
  // default label using id/type
  a.label = a.label || `${a.type} ${a.id}`;
  annotations.push(a);
  console.log('[annotations] added', a);
  renderAnnotations();
  if(window.renderAnnotationList) window.renderAnnotationList();
}

// global wrapper so code can call renderAnnotationList() safely
function renderAnnotationList(){ if(window.renderAnnotationList) window.renderAnnotationList(); }

function removeAnnotation(id){
  annotations = annotations.filter(a=>a.id!==id);
  renderAnnotations(); if(window.renderAnnotationList) window.renderAnnotationList();
}

function renderAnnotations(){
  diag('[annotations] renderAnnotations');
  if(!svgOverlay) return;
  // clear
  while(svgOverlay.firstChild) svgOverlay.removeChild(svgOverlay.firstChild);
  // draw saved annotations
  annotations.forEach(a=>{
    if(a.type==='point') drawPoint(a);
    else if(a.type==='rect') drawRect(a);
    else if(a.type==='polygon') drawPolygon(a);
  });
  // draw temporary shapes
  if(drawing && currentRectStart){
    const mouseImg = getMouseImagePoint();
    if(mouseImg){
      drawRect({type:'rect', x: Math.min(currentRectStart.x, mouseImg.x), y: Math.min(currentRectStart.y, mouseImg.y), w: Math.abs(currentRectStart.x-mouseImg.x), h: Math.abs(currentRectStart.y-mouseImg.y)}, true);
    }
  }
  if(drawing && currentPolygonPoints && currentPolygonPoints.length){
    drawPolygon({type:'polygon', points: currentPolygonPoints}, true);
  }
}

function drawPoint(a, temp=false){
  const p = imageToSVGPoint({x:a.x, y:a.y});
  const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('cx', Math.round(p.x)); c.setAttribute('cy', Math.round(p.y)); c.setAttribute('r', temp?4:6);
  c.setAttribute('fill', temp? 'rgba(255,0,0,0.6)' : 'rgba(255,0,0,0.9)');
  c.setAttribute('vector-effect', 'non-scaling-stroke');
  svgOverlay.appendChild(c);
  if(!temp){
    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x', Math.round(p.x + 8)); t.setAttribute('y', Math.round(p.y - 8));
    t.setAttribute('fill', '#000'); t.setAttribute('stroke', '#fff'); t.setAttribute('stroke-width', 0.8);
    t.setAttribute('font-size', 12); t.setAttribute('pointer-events', 'none');
    t.textContent = a.label || a.id;
    svgOverlay.appendChild(t);
  }
}
function drawRect(a, temp=false){
  // get top-left and bottom-right SVG points
  const p1 = imageToSVGPoint({x:a.x, y:a.y});
  const p2 = imageToSVGPoint({x:a.x + a.w, y:a.y + a.h});
  const x = Math.min(p1.x, p2.x), y = Math.min(p1.y, p2.y);
  const w = Math.abs(p2.x - p1.x), h = Math.abs(p2.y - p1.y);
  const r = document.createElementNS('http://www.w3.org/2000/svg','rect');
  r.setAttribute('x', Math.round(x)); r.setAttribute('y', Math.round(y)); r.setAttribute('width', Math.round(w)); r.setAttribute('height', Math.round(h));
  r.setAttribute('stroke', temp? '#f60' : '#f00'); r.setAttribute('fill', 'none'); r.setAttribute('stroke-width', temp?1.5:2);
  r.setAttribute('vector-effect', 'non-scaling-stroke');
  svgOverlay.appendChild(r);
  if(!temp){
    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x', Math.round(x + 4)); t.setAttribute('y', Math.max(12, Math.round(y - 4)));
    t.setAttribute('fill', '#000'); t.setAttribute('stroke', '#fff'); t.setAttribute('stroke-width', 0.8);
    t.setAttribute('font-size', 12); t.setAttribute('pointer-events', 'none');
    t.textContent = a.label || a.id;
    svgOverlay.appendChild(t);
  }
}
function drawPolygon(a, temp=false){
  if(!a.points || a.points.length===0) return;
  const mapped = a.points.map(pt => imageToSVGPoint(pt));
  const pts = mapped.map(v => `${v.x},${v.y}`).join(' ');
  const poly = document.createElementNS('http://www.w3.org/2000/svg','polygon');
  poly.setAttribute('points', pts);
  poly.setAttribute('stroke', temp? '#f80' : '#f00'); poly.setAttribute('fill', temp? 'rgba(255,160,0,0.2)' : 'rgba(255,0,0,0.15)'); poly.setAttribute('stroke-width', temp?1.5:2);
  poly.setAttribute('vector-effect', 'non-scaling-stroke');
  svgOverlay.appendChild(poly);

  if(!temp){
    // compute centroid for label placement
    let cx=0, cy=0; mapped.forEach(p=>{ cx += p.x; cy += p.y; }); cx /= mapped.length; cy /= mapped.length;
    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x', Math.round(cx)); t.setAttribute('y', Math.round(cy));
    t.setAttribute('fill', '#000'); t.setAttribute('stroke', '#fff'); t.setAttribute('stroke-width', 0.8);
    t.setAttribute('font-size', 12); t.setAttribute('pointer-events', 'none');
    t.textContent = a.label || a.id;
    svgOverlay.appendChild(t);
  }
}

function imageToViewerElementPoint(imgPt){
  try{
    if(!viewer || !viewer.viewport) return {x:0,y:0};
    const viewportPt = viewer.viewport.imageToViewportCoordinates(new OpenSeadragon.Point(imgPt.x, imgPt.y));
    const viewerPx = viewer.viewport.viewportToViewerElementCoordinates(viewportPt);
    return {x: viewerPx.x, y: viewerPx.y};
  }catch(e){ return {x:0,y:0}; }
}

// Convert image coordinates to SVG overlay coordinates robustly.
// This computes the image -> viewport -> viewer-element pixel coordinates, converts to client (screen)
// coordinates, then maps into the SVG overlay's coordinate system using the overlay's screen CTM.
// This ensures coordinates remain aligned even during OpenSeadragon animations/transforms.
function imageToSVGPoint(imgPt){
  try{
    if(!viewer || !viewer.viewport || !viewer.container) return {x:0,y:0};
    // image -> viewport -> viewer-element pixels
    const viewportPt = viewer.viewport.imageToViewportCoordinates(new OpenSeadragon.Point(imgPt.x, imgPt.y));
    const viewerPx = viewer.viewport.viewportToViewerElementCoordinates(viewportPt);
    const rect = viewer.container.getBoundingClientRect();
    const clientX = rect.left + viewerPx.x;
    const clientY = rect.top + viewerPx.y;

    // If SVG overlay present, use its CTM to convert screen coords into SVG coords
    if(svgOverlay && typeof svgOverlay.getScreenCTM === 'function'){
      const pt = svgOverlay.createSVGPoint(); pt.x = clientX; pt.y = clientY;
      const svgP = pt.matrixTransform(svgOverlay.getScreenCTM().inverse());
      return {x: svgP.x, y: svgP.y};
    }
    // fallback: return viewer-element pixel coords (relative to container)
    return {x: viewerPx.x, y: viewerPx.y};
  }catch(e){ console.warn('[annotations] imageToSVGPoint error', e); return {x:0,y:0}; }
}
function viewerElementToImagePoint(clientX, clientY){
  try{
    if(!viewer || !viewer.viewport || !viewer.container) return null;
    // Always use main viewer container as the reference frame for clicks (avoid influence from overlay transforms)
    const rect = viewer.container.getBoundingClientRect();
    const px = clientX - rect.left; const py = clientY - rect.top;
    const vpPt = viewer.viewport.viewerElementToViewportCoordinates(new OpenSeadragon.Point(px, py));
    const imgPt = viewer.viewport.viewportToImageCoordinates(vpPt);
    if(!imgPt || Number.isNaN(imgPt.x) || Number.isNaN(imgPt.y)) return null;
    return {x: imgPt.x, y: imgPt.y};
  }catch(e){ return null; }
}

// helper to get current mouse image coords from last mouse event on overlay
function getMouseImagePoint(){ if(!viewer) return null; if(!_lastMouseEvent) return null; return viewerElementToImagePoint(_lastMouseEvent.clientX, _lastMouseEvent.clientY); }

function saveAnnotations(){
  if(!slideName){ alert('Open a slide first'); return; }
  const data = { slide: slideName, annotations };
  const blob = new Blob([JSON.stringify(data,null,2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `annotations_${slideName}.json`;
  document.body.appendChild(a); a.click(); a.remove();
}

function loadAnnotationsFromFile(ev){
  const f = ev.target.files[0]; if(!f) return;
  const reader = new FileReader();
  reader.onload = function(e){
    try{
      const obj = JSON.parse(e.target.result);
      if(obj.slide && obj.slide!==slideName){ if(!confirm(`Loaded annotations for ${obj.slide}. Apply to current slide ${slideName}?`)) return; }
      annotations = obj.annotations || [];
      annIdCounter = annotations.reduce((m,a)=> Math.max(m, parseInt(a.id && a.id.replace(/^a/,'')) || 0), 0) + 1;
      renderAnnotations(); renderAnnotationList();
    }catch(err){ alert('Invalid JSON: ' + err); }
  }
  reader.readAsText(f);
}

// initialize overlay when viewer opens
// Annotation overlay auto-init disabled (this UI build removes annotations)

// Authentication disabled (no login wall)

// Initialize overlay controls
document.addEventListener('DOMContentLoaded', ()=>{
  try{ initOverlayControls(); }catch(e){ console.error('[overlay] initOverlayControls failed', e); }
});
// try to initialize immediately as script may run after DOMContentLoaded
try{ initOverlayControls(); }catch(e){}


