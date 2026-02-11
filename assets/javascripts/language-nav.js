// 动态显示/隐藏导航项根据当前页面语言
document.addEventListener('DOMContentLoaded', function() {
    const currentPath = window.location.pathname;
    const isChinesePage = currentPath.includes('/zh/');
    
    // 获取导航菜单
    const nav = document.querySelector('.md-nav--primary .md-nav__list');
    if (!nav) return;
    
    // 获取所有导航项
    const navItems = nav.querySelectorAll('.md-nav__item');
    
    navItems.forEach(function(item) {
        const link = item.querySelector('.md-nav__link');
        if (!link) return;
        
        const text = link.textContent.trim();
        
        if (isChinesePage) {
            // 在中文页面，隐藏英文导航项
            if (text === 'Home' || text === 'Course Notes') {
                item.style.display = 'none';
            }
        } else {
            // 在英文页面，隐藏中文导航项
            if (text === '中文笔记') {
                item.style.display = 'none';
            }
        }
    });
});
