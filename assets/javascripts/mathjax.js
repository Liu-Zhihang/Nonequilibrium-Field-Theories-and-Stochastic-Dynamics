window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams"
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre"],
    processHtmlClass: "arithmatex"
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      renderPageEnhancements();
    }
  }
};

function normalizeArithmatex() {
  document.querySelectorAll("span.arithmatex span.arithmatex").forEach((inner) => {
    const outer = inner.parentElement;
    if (outer && outer.classList.contains("arithmatex")) {
      outer.replaceWith(inner);
    }
  });
}

function promoteDisplayMath() {
  document.querySelectorAll("span.arithmatex").forEach((span) => {
    const parent = span.parentElement;
    if (!parent) {
      return;
    }
    const content = span.textContent ? span.textContent.trim() : "";
    if (!content.startsWith("\\(") || !content.endsWith("\\)")) {
      return;
    }
    const hasSiblings = Array.from(parent.childNodes).some((node) => {
      if (node === span) {
        return false;
      }
      if (node.nodeType === Node.TEXT_NODE) {
        return Boolean(node.textContent.trim());
      }
      if (node.nodeType === Node.ELEMENT_NODE) {
        return true;
      }
      return false;
    });
    if (hasSiblings) {
      return;
    }
    const display = content.slice(2, -2).trim();
    if (!display) {
      return;
    }
    const block = document.createElement("div");
    block.className = "arithmatex";
    block.textContent = `\\[${display}\\]`;
    if (parent.tagName === "P") {
      parent.replaceWith(block);
    } else {
      parent.replaceChild(block, span);
    }
  });
}

function expandActiveToc() {
  const tocToggle = document.querySelector("input.md-nav__toggle#__toc");
  if (tocToggle) {
    tocToggle.checked = true;
  }
}

function buildToc() {
  const tocList = document.querySelector(
    'nav.md-nav--secondary ul.md-nav__list[data-md-component="toc"]'
  );
  const article = document.querySelector("article");
  if (!tocList || !article) {
    return;
  }

  const groups = [];
  article.querySelectorAll("h2, h3").forEach((heading) => {
    if (!heading.id) {
      return;
    }
    if (heading.tagName === "H2") {
      groups.push({ heading, children: [] });
    } else if (heading.tagName === "H3" && groups.length) {
      groups[groups.length - 1].children.push(heading);
    }
  });

  if (!groups.length) {
    return;
  }

  tocList.innerHTML = "";
  let sectionIndex = 0;

  const createLink = (heading) => {
    const link = document.createElement("a");
    link.className = "md-nav__link";
    link.href = `#${heading.id}`;
    const span = document.createElement("span");
    span.className = "md-ellipsis";
    span.textContent = heading.textContent.trim();
    link.appendChild(span);
    return link;
  };

  groups.forEach(({ heading, children }) => {
    const text = heading.textContent.trim();

    if (!children.length) {
      const item = document.createElement("li");
      item.className = "md-nav__item";
      item.appendChild(createLink(heading));
      tocList.appendChild(item);
      return;
    }

    const item = document.createElement("li");
    item.className = "md-nav__item md-nav__item--nested";

    const toggle = document.createElement("input");
    toggle.className = "md-nav__toggle md-toggle";
    toggle.type = "checkbox";
    toggle.id = `__toc_${sectionIndex++}`;
    toggle.checked = true;
    item.appendChild(toggle);

    const label = document.createElement("label");
    label.className = "md-nav__link";
    label.setAttribute("for", toggle.id);
    const labelSpan = document.createElement("span");
    labelSpan.className = "md-ellipsis";
    labelSpan.textContent = text;
    label.appendChild(labelSpan);
    const labelIcon = document.createElement("span");
    labelIcon.className = "md-nav__icon md-icon";
    label.appendChild(labelIcon);
    item.appendChild(label);

    const nestedNav = document.createElement("nav");
    nestedNav.className = "md-nav";
    const nestedList = document.createElement("ul");
    nestedList.className = "md-nav__list";

    const topLinkItem = document.createElement("li");
    topLinkItem.className = "md-nav__item";
    topLinkItem.appendChild(createLink(heading));
    nestedList.appendChild(topLinkItem);

    children.forEach((child) => {
      const childItem = document.createElement("li");
      childItem.className = "md-nav__item";
      childItem.appendChild(createLink(child));
      nestedList.appendChild(childItem);
    });

    nestedNav.appendChild(nestedList);
    item.appendChild(nestedNav);
    tocList.appendChild(item);
  });
}

function scheduleTocRebuild() {
  requestAnimationFrame(() => {
    requestAnimationFrame(buildToc);
  });
}

function scheduleMathReflow(attempt = 0) {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise().then(() => {
      normalizeArithmatex();
      promoteDisplayMath();
      expandActiveToc();
      scheduleTocRebuild();
    });
  } else if (attempt < 10) {
    setTimeout(() => scheduleMathReflow(attempt + 1), 200);
  }
}

function renderPageEnhancements() {
  normalizeArithmatex();
  promoteDisplayMath();
  expandActiveToc();
  scheduleTocRebuild();
  scheduleMathReflow();
}

document.addEventListener("DOMContentLoaded", renderPageEnhancements);
window.addEventListener("load", renderPageEnhancements);

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    renderPageEnhancements();
  });
}
*** End Patch
