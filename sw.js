// Service worker minimal: tidak menyimpan cache, hanya memenuhi syarat agar aplikasi bisa dipasang.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", () => {});
