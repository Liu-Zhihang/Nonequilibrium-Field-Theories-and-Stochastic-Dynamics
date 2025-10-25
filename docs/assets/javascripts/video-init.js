/*
  Enhance video playback in MkDocs pages:
  - Convert Markdown images mistakenly pointing to .mp4 into <video> tags
  - Add controls/preload/playsinline to existing <video>
  - Wrap videos in a responsive container
*/
(function () {
  function wrapResponsive(el) {
    if (el.closest && el.closest('.video-container')) return;
    var wrapper = document.createElement('div');
    wrapper.className = 'video-container';
    var parent = el.parentNode;
    if (!parent) return;
    parent.insertBefore(wrapper, el);
    wrapper.appendChild(el);
  }

  function enhanceVideo(el, title) {
    if (!el.hasAttribute('controls')) el.setAttribute('controls', '');
    if (!el.hasAttribute('preload')) el.setAttribute('preload', 'metadata');
    el.setAttribute('playsinline', '');
    if (title && !el.getAttribute('title')) el.setAttribute('title', title);
    wrapResponsive(el);
  }

  function replaceImgWithVideo(img) {
    var src = img.getAttribute('src');
    if (!src) return;
    var alt = img.getAttribute('alt') || '';
    var video = document.createElement('video');
    video.src = src;
    enhanceVideo(video, alt);
    img.replaceWith(video);
  }

  function run() {
    // Convert images linking to mp4 into playable videos
    var mp4Imgs = document.querySelectorAll('img[src$=".mp4" i]');
    mp4Imgs.forEach(function (img) { replaceImgWithVideo(img); });

    // Enhance existing <video> tags
    var videos = document.querySelectorAll('video');
    videos.forEach(function (v) { enhanceVideo(v); });
  }

  // Support MkDocs Material instant loading via document$ hook
  if (window.document$ && typeof window.document$.subscribe === 'function') {
    window.document$.subscribe(run);
  } else if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();
