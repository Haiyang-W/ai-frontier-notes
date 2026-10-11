# 学习笔记网站

当前版本只发布空的首页、归档和关于页面，不读取或复制仓库中的笔记。

网站地址：https://haiyang-w.github.io/ai-frontier-notes/

本地预览（从仓库根目录运行）：

```sh
python3 -m http.server 8000 --directory site
```

打开 http://localhost:8000/ 。

部署：在仓库 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。推送 `site/` 的改动到 `main` 后，`Deploy notes site` 工作流会自动发布；也可以从 Actions 页面手动运行。网站使用相对路径，可以在当前项目路径下部署，并且不会替换学术主页。

- `index.html`：空的笔记首页
- `archive.html`：空的归档页
- `about.html`：简短的网站介绍
- `styles.css`：中文排版、手机布局与深浅色样式
- `theme.js`：主题切换与偏好保存

后续接入文章时，可以保持这些阅读样式，再增加静态生成器或文章页面。
