/* Lidong Yu — site behaviour. No dependencies. */
(function () {
  'use strict';

  var doc = document;
  var root = doc.documentElement;
  root.parentNode.classList.remove('no-js');

  /* ---------- Theme ---------- */
  var THEME_KEY = 'ldy-theme';
  function applyTheme(t) {
    if (t === 'light' || t === 'dark') { root.setAttribute('data-theme', t); }
    else { root.removeAttribute('data-theme'); }
    var m = doc.querySelector('meta[name="theme-color"]');
    if (m) {
      var dark = t === 'dark' || (!t && window.matchMedia('(prefers-color-scheme: dark)').matches);
      m.setAttribute('content', dark ? '#08090c' : '#ffffff');
    }
  }
  try { applyTheme(localStorage.getItem(THEME_KEY)); } catch (e) { applyTheme(null); }

  function syncThemeBtn() {
    var btn = doc.querySelector('[data-theme-toggle]');
    if (!btn) return;
    var isDark = root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    btn.setAttribute('aria-pressed', isDark ? 'true' : 'false');
    btn.setAttribute('aria-label', isDark ? (btn.dataset.labelLight || 'Switch to light theme')
                                          : (btn.dataset.labelDark || 'Switch to dark theme'));
    btn.title = btn.getAttribute('aria-label');
  }
  applyTheme(root.getAttribute('data-theme'));
  syncThemeBtn();

  doc.addEventListener('click', function (e) {
    var t = e.target.closest && e.target.closest('[data-theme-toggle]');
    if (t) {
      var isDark = root.getAttribute('data-theme') === 'dark' ||
        (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
      var next = isDark ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem(THEME_KEY, next); } catch (err) {}
      syncThemeBtn();
      return;
    }
    var langBtn = e.target.closest && e.target.closest('[data-lang-toggle]');
    if (langBtn) {
      var alt = root.getAttribute('data-alt');
      if (alt) window.location.href = alt;
    }
  });

  /* ---------- Mobile nav ---------- */
  var toggle = doc.querySelector('.nav-toggle');
  var nav = doc.getElementById('site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      doc.body.classList.toggle('nav-open', open);
    });
    nav.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
        doc.body.classList.remove('nav-open');
      }
    });
    doc.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && nav.classList.contains('is-open')) {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
        doc.body.classList.remove('nav-open');
      }
    });
  }

  /* ---------- Sticky header + back to top ---------- */
  var header = doc.querySelector('.site-header');
  var toTop = doc.querySelector('.to-top');
  var raf = false;
  function onScroll() {
    var y = window.scrollY || doc.documentElement.scrollTop;
    if (header) header.classList.toggle('is-stuck', y > 8);
    if (toTop) toTop.classList.toggle('is-visible', y > 700);
    var bar = doc.querySelector('.reading-bar');
    if (bar) {
      var h = doc.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = (h > 0 ? Math.min(100, (y / h) * 100) : 0) + '%';
    }
    raf = false;
  }
  window.addEventListener('scroll', function () {
    if (!raf) { raf = true; window.requestAnimationFrame(onScroll); }
  }, { passive: true });
  onScroll();
  if (toTop) toTop.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  /* ---------- Scroll reveal ---------- */
  var reveals = doc.querySelectorAll('[data-reveal]');
  if ('IntersectionObserver' in window && reveals.length) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('is-in'); io.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.04 });
    reveals.forEach(function (el) { io.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add('is-in'); });
  }

  /* ---------- Repo filters ---------- */
  var filters = doc.querySelectorAll('.chip[data-filter]');
  if (filters.length) {
    filters.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var want = btn.getAttribute('data-filter');
        filters.forEach(function (b) {
          b.setAttribute('aria-pressed', b === btn ? 'true' : 'false');
        });
        doc.querySelectorAll('.repo[data-kind]').forEach(function (card) {
          var kinds = (card.getAttribute('data-kind') || '').split(/\s+/);
          var show = want === 'all' || kinds.indexOf(want) !== -1;
          card.hidden = !show;
        });
        doc.querySelectorAll('[data-group]').forEach(function (group) {
          var kids = group.querySelectorAll('.repo[data-kind]');
          var visible = 0;
          kids.forEach(function (k) { if (!k.hidden) visible++; });
          group.hidden = visible === 0;
        });
      });
    });
  }

  /* ---------- Copy to clipboard ---------- */
  var toast = doc.querySelector('.toast');
  var toastTimer = null;
  doc.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('[data-copy]');
    if (!btn) return;
    var text = btn.getAttribute('data-copy');
    function done(ok) {
      if (!toast) return;
      toast.textContent = ok ? (toast.dataset.copied || 'Copied') : (toast.dataset.failed || 'Copy failed');
      toast.classList.add('is-on');
      clearTimeout(toastTimer);
      toastTimer = setTimeout(function () { toast.classList.remove('is-on'); }, 1800);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
    } else {
      try {
        var ta = doc.createElement('textarea');
        ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
        doc.body.appendChild(ta); ta.select(); doc.execCommand('copy'); doc.body.removeChild(ta);
        done(true);
      } catch (err) { done(false); }
    }
  });

  /* ---------- Year ---------- */
  doc.querySelectorAll('[data-year]').forEach(function (el) {
    el.textContent = String(new Date().getFullYear());
  });
})();
