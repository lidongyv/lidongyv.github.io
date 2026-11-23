// planning-architecture 子页共享脚本：进场动画 + 锚点平滑滚动。零依赖。
(function () {
  "use strict";
  document.documentElement.classList.add("js");
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); } });
  }, { threshold: 0.06 });
  document.querySelectorAll("section, .panel, .opt-item, .pt, .stat, .acard, .step").forEach(function (el) {
    el.classList.add("reveal"); io.observe(el);
  });
  document.querySelectorAll('a[href^="#"]').forEach(function (a) {
    a.addEventListener("click", function (ev) {
      var id = a.getAttribute("href");
      if (id && id.length > 1) {
        var t = document.querySelector(id);
        if (t) { ev.preventDefault(); t.scrollIntoView({ behavior: "smooth", block: "start" }); }
      }
    });
  });
})();
