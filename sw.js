/**
 * Service worker — exists mainly so the app is installable as a PWA.
 * Strategy: network-first for the app shell (so a deploy is picked up on the
 * next online visit), cache fallback when offline. Episode audio is NEVER
 * intercepted: app.js streams it through its own MediaSource pipeline with
 * fetch timeouts tuned for locked phones — the worker must stay out of that
 * path. Bump CACHE (and the ?v= entries) on every deploy, same as APP_VERSION.
 */
const CACHE = 'dodo-v23';
const SHELL = [
  './',
  './index.html',
  './app.js?v=23',
  './style.css?v=23',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      // Only reap our own outdated shell caches (dodo-vN). The page's
      // 'dodo-episodes' cache holds the downloaded episodes and must
      // survive every app update.
      .then(keys => Promise.all(
        keys.filter(k => k.startsWith('dodo-v') && k !== CACHE).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Same-origin GET shell requests only. Episode audio + index.json go
  // straight to the network (default browser handling), as do the GitHub
  // API calls used for server-side delete.
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;
  if (url.pathname.includes('/episodes/')) return;
  e.respondWith(
    fetch(e.request)
      .then(resp => {
        if (resp.ok) {
          const copy = resp.clone();
          caches.open(CACHE).then(c => c.put(e.request, copy));
        }
        return resp;
      })
      // ignoreSearch: an offline load still finds the shell even if the
      // cached ?v= differs from the requested one
      .catch(() => caches.match(e.request, { ignoreSearch: true }))
  );
});
