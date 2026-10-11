(() => {
  const root = document.documentElement;
  const button = document.querySelector('.theme-toggle');
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)');
  const isDark = () => root.dataset.theme ? root.dataset.theme === 'dark' : systemDark.matches;
  const update = () => {
    const label = isDark() ? '切换浅色模式' : '切换深色模式';
    button.setAttribute('aria-label', label);
    button.title = label;
  };
  try {
    const saved = localStorage.getItem('notes-theme');
    if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;
  } catch { /* Theme switching still works if storage is unavailable. */ }
  button.addEventListener('click', () => {
    root.dataset.theme = isDark() ? 'light' : 'dark';
    try { localStorage.setItem('notes-theme', root.dataset.theme); } catch {}
    update();
  });
  systemDark.addEventListener('change', update);
  update();
})();
