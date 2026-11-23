// GS 阅读站：论文清单页筛选逻辑（file:// 下无 fetch，数据来自 data/papers.js）
(function () {
  'use strict';
  function boot() {
    var q = document.getElementById('q');
    var fVenue = document.getElementById('fVenue');
    var fCat = document.getElementById('fCat');
    var clearBtn = document.getElementById('clear');
    var shown = document.getElementById('shown');
    var total = document.getElementById('totalCount');
    if (!q || !fVenue || !fCat) return;
    var rows = Array.prototype.slice.call(document.querySelectorAll('#paperTable tbody tr'));
    var n = rows.length;
    if (total) total.textContent = String(n);
    function apply() {
      var kw = q.value.trim().toLowerCase();
      var v = fVenue.value, c = fCat.value, count = 0;
      rows.forEach(function (tr) {
        var ok = true;
        if (v && tr.getAttribute('data-venue') !== v) ok = false;
        if (c && tr.getAttribute('data-cat') !== c) ok = false;
        if (ok && kw && tr.textContent.toLowerCase().indexOf(kw) === -1) ok = false;
        tr.classList.toggle('hidden', !ok);
        if (ok) count++;
      });
      if (shown) shown.textContent = '显示 ' + count + ' / ' + n;
    }
    q.addEventListener('input', apply);
    fVenue.addEventListener('change', apply);
    fCat.addEventListener('change', apply);
    if (clearBtn) {
      clearBtn.addEventListener('click', function () {
        q.value = ''; fVenue.value = ''; fCat.value = ''; apply();
      });
    }
    apply();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
