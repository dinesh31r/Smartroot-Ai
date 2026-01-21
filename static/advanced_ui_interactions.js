/* ============================================================
🌱 SMARTROOT-AI — ADVANCED UI INTERACTIONS & FEATURES
============================================================ */

// ---- DOM READY ----
document.addEventListener('DOMContentLoaded', function () {
  initializeAdvancedUI();
  setupInteractiveElements();
  setupScrollAnimations();
  setupCopyToClipboard();
});

// Theme toggle is handled by advanced_ui_overlay.html
// Disabled to prevent conflicts with the primary theme implementation
// document.addEventListener('DOMContentLoaded', function() {
//   setTimeout(function() {
//     setupThemeToggle();
//   }, 500);
// });

// ---- INITIALIZE ADVANCED UI ----
function initializeAdvancedUI() {
  // Add animation classes to elements
  const metrics = document.querySelectorAll('[data-testid="metric-container"]');
  const boxes = document.querySelectorAll('[data-testid="info-box"], [data-testid="warning-box"], [data-testid="success-box"]');
  const expanders = document.querySelectorAll('[data-testid="expander"]');

  metrics.forEach((metric, index) => {
    metric.classList.add('floating-card');
    metric.style.animationDelay = `${index * 0.1}s`;
  });

  boxes.forEach((box, index) => {
    box.classList.add('smooth-transition');
    box.style.animationDelay = `${index * 0.1}s`;
  });

  expanders.forEach((exp, index) => {
    exp.classList.add('smooth-transition');
    exp.style.animationDelay = `${index * 0.1}s`;
  });

  // Theme toggle is now handled by advanced_ui_overlay.html
  // setupThemeToggle();

  // Setup keyboard shortcuts
  setupKeyboardShortcuts();
}

// ---- THEME TOGGLE ----
function setupThemeToggle() {
  // Get saved theme or detect system preference
  const savedTheme = localStorage.getItem('smartroot-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');

  // Apply initial theme
  applyTheme(initialTheme);

  // Setup event listeners
  const themeLight = document.getElementById('themeLight');
  const themeDark = document.getElementById('themeDark');

  if (themeLight) {
    themeLight.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      applyTheme('light');
      localStorage.setItem('smartroot-theme', 'light');
    });
  }

  if (themeDark) {
    themeDark.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      applyTheme('dark');
      localStorage.setItem('smartroot-theme', 'dark');
    });
  }
}

// Apply theme and ensure text visibility
function applyTheme(theme) {
  const themeLight = document.getElementById('themeLight');
  const themeDark = document.getElementById('themeDark');

  // Update button states
  if (themeLight && themeDark) {
    themeLight.classList.toggle('active', theme === 'light');
    themeDark.classList.toggle('active', theme === 'dark');
  }

  // Set color scheme
  if (theme === 'dark') {
    document.documentElement.style.colorScheme = 'dark';
    // Ensure text is readable in dark mode
    document.body.style.color = '#e5e7eb';
    document.documentElement.style.color = '#e5e7eb';
  } else {
    document.documentElement.style.colorScheme = 'light';
    // Ensure text is readable in light mode
    document.body.style.color = '#1f2937';
    document.documentElement.style.color = '#1f2937';
  }
}

// ---- SETUP INTERACTIVE ELEMENTS ----
function setupInteractiveElements() {
  // Add ripple effect to buttons
  const buttons = document.querySelectorAll('button');
  buttons.forEach(button => {
    button.addEventListener('click', function (e) {
      const ripple = document.createElement('span');
      const rect = this.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;

      ripple.style.width = ripple.style.height = size + 'px';
      ripple.style.left = x + 'px';
      ripple.style.top = y + 'px';

      this.appendChild(ripple);

      setTimeout(() => ripple.remove(), 600);
    });
  });

  // Add hover tooltips to input fields
  const inputs = document.querySelectorAll('input, select, textarea');
  inputs.forEach(input => {
    input.addEventListener('focus', function () {
      this.style.transform = 'scale(1.02)';
    });

    input.addEventListener('blur', function () {
      this.style.transform = 'scale(1)';
    });
  });
}

// ---- SCROLL ANIMATIONS ----
function setupScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  });

  // Observe all metric containers and boxes
  const elements = document.querySelectorAll(
    '[data-testid="metric-container"], [data-testid="info-box"], [data-testid="warning-box"]'
  );

  elements.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });
}

// ---- COPY TO CLIPBOARD FUNCTIONALITY ----
function setupCopyToClipboard() {
  const codes = document.querySelectorAll('code');
  codes.forEach(code => {
    code.style.cursor = 'pointer';
    code.title = 'Click to copy';

    code.addEventListener('click', function (e) {
      e.stopPropagation();
      const text = this.textContent;

      navigator.clipboard.writeText(text).then(() => {
        const originalText = this.textContent;
        this.textContent = '✓ Copied!';
        this.style.background = 'rgba(16, 185, 129, 0.2)';

        setTimeout(() => {
          this.textContent = originalText;
          this.style.background = '';
        }, 2000);
      });
    });
  });
}

// ---- KEYBOARD SHORTCUTS ----
function setupKeyboardShortcuts() {
  document.addEventListener('keydown', function (e) {
    // Ctrl/Cmd + K: Focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      const searchInput = document.querySelector('input[type="search"]');
      if (searchInput) searchInput.focus();
    }

    // Ctrl/Cmd + L: Focus upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
      e.preventDefault();
      const uploadInput = document.querySelector('input[type="file"]');
      if (uploadInput) uploadInput.focus();
    }
  });
}

// ---- DYNAMIC METRIC COUNTER ----
class MetricCounter {
  constructor(element, target) {
    this.element = element;
    this.target = target;
    this.current = 0;
    this.speed = 30;
  }

  animate() {
    const increment = this.target / this.speed;
    const counter = setInterval(() => {
      this.current += increment;
      if (this.current >= this.target) {
        this.current = this.target;
        clearInterval(counter);
      }
      this.element.textContent = Math.round(this.current) + '%';
    }, 10);
  }
}

// ---- FLOATING PARTICLES (BACKGROUND) ----
function createFloatingParticles() {
  const canvas = document.getElementById('particles-js');
  if (!canvas) return;

  const particles = [];
  const particleCount = 30;

  class Particle {
    constructor() {
      this.x = Math.random() * window.innerWidth;
      this.y = Math.random() * window.innerHeight;
      this.size = Math.random() * 3 + 1;
      this.speedX = Math.random() * 0.5 - 0.25;
      this.speedY = Math.random() * 0.5 - 0.25;
      this.opacity = Math.random() * 0.5 + 0.2;
    }

    update() {
      this.x += this.speedX;
      this.y += this.speedY;

      if (this.x > window.innerWidth) this.x = 0;
      if (this.x < 0) this.x = window.innerWidth;
      if (this.y > window.innerHeight) this.y = 0;
      if (this.y < 0) this.y = window.innerHeight;
    }

    draw(ctx) {
      ctx.fillStyle = `rgba(16, 185, 129, ${this.opacity})`;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (canvas && canvas.tagName === 'CANVAS') {
    const ctx = canvas.getContext('2d');

    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach(particle => {
        particle.update();
        particle.draw(ctx);
      });

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 150) {
            ctx.strokeStyle = `rgba(16, 185, 129, ${0.2 * (1 - distance / 150)})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      requestAnimationFrame(animate);
    }

    animate();
  }
}

// ---- SMOOTH SCROLL ----
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ---- LAZY LOAD IMAGES ----
if ('IntersectionObserver' in window) {
  const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        if (img.dataset.src) {
          img.src = img.dataset.src;
          img.classList.add('loaded');
        }
        observer.unobserve(img);
      }
    });
  });

  document.querySelectorAll('img[data-src]').forEach(img => {
    imageObserver.observe(img);
  });
}

// ---- UTILITY: FORMAT LARGE NUMBERS ----
function formatNumber(num) {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toString();
}

// ---- UTILITY: DEBOUNCE ----
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// ---- UTILITY: THROTTLE ----
function throttle(func, limit) {
  let inThrottle;
  return function (...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// ---- WINDOW RESIZE HANDLER ----
window.addEventListener('resize', debounce(function () {
  // Handle responsive adjustments
  const width = window.innerWidth;
  if (width < 768) {
    document.querySelectorAll('[data-testid="column"]').forEach(col => {
      col.style.marginBottom = '1rem';
    });
  }
}, 250));

// ---- SHOW LOADING SPINNER ----
function showLoadingSpinner(message = 'Loading...') {
  const spinner = document.createElement('div');
  spinner.className = 'pulse-loader';
  spinner.id = 'loading-spinner';
  spinner.title = message;
  document.body.appendChild(spinner);

  spinner.style.position = 'fixed';
  spinner.style.top = '50%';
  spinner.style.left = '50%';
  spinner.style.transform = 'translate(-50%, -50%)';
  spinner.style.zIndex = '9999';
}

function hideLoadingSpinner() {
  const spinner = document.getElementById('loading-spinner');
  if (spinner) {
    spinner.style.opacity = '0';
    spinner.style.transform = 'translate(-50%, -50%) scale(0.8)';
    spinner.style.transition = 'all 0.3s ease';
    setTimeout(() => spinner.remove(), 300);
  }
}

// ---- TOAST NOTIFICATIONS ----
function showToast(message, type = 'info', duration = 3000) {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  toast.style.position = 'fixed';
  toast.style.bottom = '2rem';
  toast.style.right = '2rem';
  toast.style.padding = '1rem 1.5rem';
  toast.style.borderRadius = '0.75rem';
  toast.style.zIndex = '10000';
  toast.style.animation = 'slideInRight 0.3s ease-out';
  toast.style.fontWeight = '500';

  if (type === 'success') {
    toast.style.background = 'linear-gradient(135deg, #10b981, #06b6d4)';
    toast.style.color = 'white';
  } else if (type === 'error') {
    toast.style.background = '#ef4444';
    toast.style.color = 'white';
  } else {
    toast.style.background = '#3b82f6';
    toast.style.color = 'white';
  }

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(400px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ---- EXPORT UTILITIES ----
window.SmartRootUI = {
  showLoadingSpinner,
  hideLoadingSpinner,
  showToast,
  formatNumber,
  debounce,
  throttle,
  MetricCounter
};

// ---- INTERACTIVE METRIC CARDS ----
function setupInteractiveMetrics() {
  const metricCards = document.querySelectorAll('.metric-card');

  metricCards.forEach(card => {
    card.addEventListener('click', function () {
      this.style.transform = 'scale(1.05)';
      setTimeout(() => {
        this.style.transform = '';
      }, 300);
    });

    card.addEventListener('mouseenter', function () {
      const parent = this.closest('.metrics-dashboard');
      if (parent) {
        parent.querySelectorAll('.metric-card').forEach(sibling => {
          if (sibling !== this) {
            sibling.style.opacity = '0.7';
            sibling.style.filter = 'blur(0.5px)';
          }
        });
      }
    });

    card.addEventListener('mouseleave', function () {
      const parent = this.closest('.metrics-dashboard');
      if (parent) {
        parent.querySelectorAll('.metric-card').forEach(sibling => {
          sibling.style.opacity = '1';
          sibling.style.filter = 'blur(0px)';
        });
      }
    });
  });
}

// ---- TOOLTIP INITIALIZATION ----
function setupTooltips() {
  const tooltipIcons = document.querySelectorAll('.tooltip-icon');

  tooltipIcons.forEach(icon => {
    icon.addEventListener('mouseover', function () {
      this.style.zIndex = '1000';
    });

    icon.addEventListener('mouseout', function () {
      this.style.zIndex = 'auto';
    });
  });
}

// ---- ANIMATED GAUGE BARS ----
function animateGaugeBars() {
  const gaugeFills = document.querySelectorAll('.gauge-fill');

  gaugeFills.forEach(fill => {
    const width = fill.style.width;
    fill.style.width = '0%';

    setTimeout(() => {
      fill.style.width = width;
    }, 100);
  });
}

// ---- ACTION ITEMS INTERACTION ----
function setupActionItems() {
  const actionItems = document.querySelectorAll('.action-item');

  actionItems.forEach((item, index) => {
    item.style.animationDelay = `${index * 0.1}s`;
    item.style.opacity = '0';
    item.style.animation = 'slideIn 0.4s ease-out forwards';

    item.addEventListener('mouseenter', function () {
      this.style.boxShadow = '0 8px 24px rgba(16, 185, 129, 0.2)';
    });

    item.addEventListener('mouseleave', function () {
      this.style.boxShadow = '';
    });
  });
}

// ---- UPLOAD CONFIRMATION ANIMATION ----
function animateUploadConfirmation() {
  const uploadInput = document.querySelector('input[type="file"]');

  if (uploadInput) {
    uploadInput.addEventListener('change', function () {
      if (this.files.length > 0) {
        const fileName = this.files[0].name;

        // Show confirmation
        const existingMsg = document.querySelector('[data-upload-confirmation]');
        if (existingMsg) existingMsg.remove();

        const confirmationMsg = document.createElement('div');
        confirmationMsg.setAttribute('data-upload-confirmation', 'true');
        confirmationMsg.style.cssText = `
          margin-top: 1rem;
          padding: 1rem;
          background: rgba(16, 185, 129, 0.1);
          border: 2px solid #10b981;
          border-radius: 0.75rem;
          color: #10b981;
          font-weight: 600;
          animation: slideIn 0.3s ease-out;
        `;
        confirmationMsg.innerHTML = `✓ File "${fileName}" uploaded successfully!`;

        this.parentElement.appendChild(confirmationMsg);

        setTimeout(() => {
          confirmationMsg.style.opacity = '0';
          confirmationMsg.style.transition = 'opacity 0.3s ease';
          setTimeout(() => confirmationMsg.remove(), 300);
        }, 3000);
      }
    });
  }
}

// ---- COMPARISON CONTROL SETUP ----
function setupComparisonControls() {
  const ctaButtons = document.querySelectorAll('.cta-primary, .cta-secondary');

  ctaButtons.forEach(button => {
    button.addEventListener('mouseover', function () {
      this.style.transform = this.classList.contains('cta-primary') ?
        'translateY(-3px)' : 'translateY(-2px)';
    });

    button.addEventListener('mouseout', function () {
      this.style.transform = 'translateY(0)';
    });
  });
}

// ---- EXPORT FUNCTIONALITY ----
function setupExportOptions() {
  // Listen for download button clicks
  const downloadButtons = document.querySelectorAll('[data-testid="stDownloadButton"]');

  downloadButtons.forEach(btn => {
    btn.addEventListener('click', function () {
      SmartRootUI.showToast('Downloading analysis results...', 'success', 2000);
    });
  });
}

// ---- STRESS METRICS HORIZONTAL ALIGNMENT ----
function enhanceStressMetricsDisplay() {
  const stressMetrics = document.querySelectorAll('.stress-metric-item');

  stressMetrics.forEach((metric, index) => {
    metric.style.animationDelay = `${index * 0.05}s`;
    metric.style.opacity = '0';
    metric.style.animation = 'slideIn 0.3s ease-out forwards';
  });
}

// ---- SECTION SEPARATOR ANIMATION ----
function animateSectionSeparators() {
  const separators = document.querySelectorAll('.section-separator');

  separators.forEach(sep => {
    sep.style.opacity = '0';

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transition = 'opacity 0.6s ease';
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    observer.observe(sep);
  });
}

// ---- TREND INDICATOR ANIMATION ----
function animateTrendIndicators() {
  const trendIndicators = document.querySelectorAll('.trend-indicator');

  trendIndicators.forEach(indicator => {
    indicator.style.opacity = '0';
    indicator.style.transform = 'scale(0.8)';

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.animation = 'scaleIn 0.4s ease-out forwards';
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    observer.observe(indicator);
  });
}

// ---- CLICKABLE DETAIL EXPANSION ----
function setupExpandableDetails() {
  const detailItems = document.querySelectorAll('[data-expandable]');

  detailItems.forEach(item => {
    item.style.cursor = 'pointer';

    item.addEventListener('click', function (e) {
      e.stopPropagation();
      const details = this.querySelector('[data-details]');

      if (details) {
        const isOpen = details.style.display !== 'none';
        details.style.display = isOpen ? 'none' : 'block';
        details.style.animation = isOpen ? '' : 'slideIn 0.3s ease-out';
      }
    });
  });
}

// ---- RECOMMENDED ACTIONS HIGHLIGHTING ----
function setupRecommendedActionsHighlight() {
  const actionItems = document.querySelectorAll('.action-item');

  actionItems.forEach(item => {
    item.addEventListener('mouseenter', function () {
      // Highlight current action
      this.style.background = 'rgba(16, 185, 129, 0.1)';
      this.style.borderLeftColor = '#10b981';

      // Dim siblings
      actionItems.forEach(sibling => {
        if (sibling !== this) {
          sibling.style.opacity = '0.6';
        }
      });
    });

    item.addEventListener('mouseleave', function () {
      // Reset all
      actionItems.forEach(sibling => {
        sibling.style.opacity = '1';
        sibling.style.background = 'rgba(255, 255, 255, 0.6)';
        sibling.style.borderLeftColor = 'var(--primary-green)';
      });
    });
  });
}

// ---- INITIALIZE ALL NEW FEATURES ----
function initializeEnhancedInteractions() {
  setupInteractiveMetrics();
  setupTooltips();
  animateGaugeBars();
  setupActionItems();
  animateUploadConfirmation();
  setupComparisonControls();
  setupExportOptions();
  enhanceStressMetricsDisplay();
  animateSectionSeparators();
  animateTrendIndicators();
  setupExpandableDetails();
  setupRecommendedActionsHighlight();
}

// ---- CALL INITIALIZATION WHEN DOM IS READY ----
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeEnhancedInteractions);
} else {
  initializeEnhancedInteractions();
}

/* ============================================================
   🚀 AUTO-INJECTED UI COMPONENTS
   ============================================================ */

// ---- QUICK ACTIONS TOOLBAR ----
function injectQuickActionsToolbar() {
  // Remove existing toolbar if any
  const existing = document.querySelector('.quick-actions-bar');
  if (existing) existing.remove();

  const toolbar = document.createElement('div');
  toolbar.className = 'quick-actions-bar';
  toolbar.innerHTML = `
    <button class="quick-action-btn" data-tooltip="Upload Image" onclick="document.querySelector('input[type=file]')?.click()">
      📷
    </button>
    <button class="quick-action-btn" data-tooltip="Scroll to Analysis" onclick="document.querySelector('#root-analysis')?.scrollIntoView({behavior:'smooth'})">
      📊
    </button>
    <div class="quick-actions-divider"></div>
    <button class="quick-action-btn" data-tooltip="Scroll to Simulate" onclick="document.querySelector('#simulate-root')?.scrollIntoView({behavior:'smooth'})">
      🌱
    </button>
    <button class="quick-action-btn" data-tooltip="Download Report" onclick="document.querySelector('[data-testid=stDownloadButton] button')?.click()">
      📥
    </button>
    <div class="quick-actions-divider"></div>
    <button class="quick-action-btn" data-tooltip="Scroll to Top" onclick="window.scrollTo({top:0,behavior:'smooth'})">
      ⬆️
    </button>
  `;

  document.body.appendChild(toolbar);
}

// ---- TOAST CONTAINER ----
function injectToastContainer() {
  if (document.querySelector('.toast-container')) return;

  const container = document.createElement('div');
  container.className = 'toast-container';
  container.id = 'toast-container';
  document.body.appendChild(container);
}

// ---- ENHANCED TOAST FUNCTION ----
function showEnhancedToast(title, message, type = 'info', duration = 5000) {
  const container = document.getElementById('toast-container') || document.body;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const icons = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ'
  };

  toast.innerHTML = `
    <div class="toast-icon">${icons[type] || icons.info}</div>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
    <div class="toast-progress">
      <div class="toast-progress-fill" style="animation-duration: ${duration}ms"></div>
    </div>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 300);
  }, duration);

  return toast;
}

// ---- INFO TOOLTIPS ON LABELS ----
function injectTooltipsOnLabels() {
  const tooltipData = {
    'Symmetry Index': 'Measures how evenly distributed the root system is. Higher values indicate more balanced growth.',
    'Water Absorption': 'Estimated efficiency of water uptake based on root density and distribution.',
    'Nutrient Uptake': 'Efficiency of nutrient absorption based on root structure and surface area.',
    'ID Confidence': 'Machine learning model confidence in species identification.',
    'Branch Density': 'Number of branch points relative to root length.',
    'Growth Direction': 'Primary direction of root growth (vertical, horizontal, or mixed).',
    'Root Age': 'Estimated age based on structural characteristics.',
    'Biomass': 'Estimated total root biomass based on area and density.',
    'Root Type': 'Classification of root system type (Fibrous, Taproot, etc.).',
    'Health Status': 'Overall health assessment based on multiple factors.',
    'Root Health Index': 'Composite score (0-100) representing overall root health.'
  };

  // Find all labels and add tooltips
  document.querySelectorAll('.analytics-label, .analytics-box .analytics-label, td:first-child').forEach(label => {
    const text = label.textContent?.trim();
    if (tooltipData[text] && !label.querySelector('.info-icon')) {
      const infoIcon = document.createElement('span');
      infoIcon.className = 'info-icon';
      infoIcon.setAttribute('data-tooltip', tooltipData[text]);
      infoIcon.innerHTML = '?';
      label.appendChild(infoIcon);
    }
  });
}

// ---- ENHANCE RESULT CARDS WITH ICONS ----
function enhanceResultCards() {
  const metricContainers = document.querySelectorAll('[data-testid="metric-container"]');

  const iconMap = {
    'symmetry': { icon: '📐', class: 'icon-blue' },
    'water': { icon: '💧', class: 'icon-blue' },
    'nutrient': { icon: '🌱', class: 'icon-green' },
    'confidence': { icon: '🔍', class: 'icon-purple' },
    'health': { icon: '🩺', class: 'icon-green' },
    'stress': { icon: '⚡', class: 'icon-orange' },
    'root': { icon: '🌿', class: 'icon-green' },
    'density': { icon: '📊', class: 'icon-blue' }
  };

  metricContainers.forEach(container => {
    if (container.dataset.enhanced) return;
    container.dataset.enhanced = 'true';

    const label = container.querySelector('label')?.textContent?.toLowerCase() || '';

    // Find matching icon
    let iconData = null;
    for (const [key, data] of Object.entries(iconMap)) {
      if (label.includes(key)) {
        iconData = data;
        break;
      }
    }

    if (iconData) {
      const iconWrapper = document.createElement('div');
      iconWrapper.className = `result-card-icon ${iconData.class}`;
      iconWrapper.innerHTML = iconData.icon;
      iconWrapper.style.cssText = 'position: absolute; top: 12px; right: 12px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; border-radius: 10px;';

      container.style.position = 'relative';
      container.appendChild(iconWrapper);
    }
  });
}

// ---- SKELETON LOADER FOR SPINNERS ----
function enhanceSpinners() {
  // Watch for Streamlit spinners and enhance them
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) {
          // Look for spinner elements
          const spinners = node.querySelectorAll ? node.querySelectorAll('[data-testid="stSpinner"]') : [];
          spinners.forEach(spinner => {
            if (!spinner.dataset.enhanced) {
              spinner.dataset.enhanced = 'true';
              spinner.innerHTML = `
                <div class="progress-indicator" style="padding: 2rem;">
                  <div class="progress-ring">
                    <div style="font-size: 1.5rem;">🌱</div>
                  </div>
                  <div class="progress-bar-container" style="width: 200px;">
                    <div class="progress-bar-fill" style="width: 60%;"></div>
                  </div>
                  <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 0.5rem;">Analyzing...</p>
                </div>
              ` + spinner.innerHTML;
            }
          });
        }
      });
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

// ---- DRAG AND DROP ENHANCEMENT ----
function enhanceDragAndDrop() {
  const dropzones = document.querySelectorAll('[data-testid="fileUploadDropzone"], [data-testid="stFileUploaderDropzone"]');

  dropzones.forEach(zone => {
    if (zone.dataset.dragEnhanced) return;
    zone.dataset.dragEnhanced = 'true';

    zone.addEventListener('dragenter', function (e) {
      e.preventDefault();
      this.classList.add('drag-active');
      this.style.transform = 'scale(1.02)';
      this.style.borderColor = '#0071e3';
    });

    zone.addEventListener('dragleave', function (e) {
      e.preventDefault();
      this.classList.remove('drag-active');
      this.style.transform = '';
      this.style.borderColor = '';
    });

    zone.addEventListener('dragover', function (e) {
      e.preventDefault();
    });

    zone.addEventListener('drop', function (e) {
      this.classList.remove('drag-active');
      this.style.transform = '';
      this.style.borderColor = '';

      // Show toast on drop
      showEnhancedToast('File Received', 'Processing your image...', 'info', 3000);
    });
  });
}

// ---- ADD GRADIENT CLASSES TO VALUES ----
function addGradientClassesToValues() {
  document.querySelectorAll('[data-testid="stMetricValue"]').forEach(value => {
    const text = value.textContent || '';
    const numMatch = text.match(/(\d+)/);

    if (numMatch) {
      const num = parseInt(numMatch[1]);
      value.classList.remove('metric-excellent', 'metric-good', 'metric-warning', 'metric-critical');

      if (num >= 80) {
        value.classList.add('metric-excellent');
      } else if (num >= 60) {
        value.classList.add('metric-good');
      } else if (num >= 40) {
        value.classList.add('metric-warning');
      } else {
        value.classList.add('metric-critical');
      }
    }
  });
}

// ---- STAGGER ANIMATION FOR SECTIONS ----
function addStaggerAnimations() {
  const sections = document.querySelectorAll('.section-card, .stress-result-box, .analytics-box, [data-testid="stExpander"]');

  sections.forEach((section, index) => {
    if (!section.dataset.animated) {
      section.dataset.animated = 'true';
      section.style.animationDelay = `${index * 0.1}s`;
    }
  });
}

// ---- INITIALIZE ALL INJECTIONS ----
function initializeUIInjections() {
  // Wait a bit for Streamlit to finish rendering
  setTimeout(() => {
    injectQuickActionsToolbar();
    injectToastContainer();
    injectTooltipsOnLabels();
    enhanceResultCards();
    enhanceSpinners();
    enhanceDragAndDrop();
    addGradientClassesToValues();
    addStaggerAnimations();

    // Show welcome toast
    showEnhancedToast('SmartRoot AI', 'Ready to analyze your plant roots!', 'success', 4000);
  }, 1000);

  // Re-run on Streamlit updates
  const observer = new MutationObserver(() => {
    injectTooltipsOnLabels();
    enhanceResultCards();
    enhanceDragAndDrop();
    addGradientClassesToValues();
    addStaggerAnimations();
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

// ---- EXPORT ENHANCED FUNCTIONS ----
window.SmartRootUI = {
  ...window.SmartRootUI,
  showEnhancedToast,
  injectQuickActionsToolbar,
  initializeUIInjections
};

/* ============================================================
   🎨 VISUAL ENHANCEMENT IMPLEMENTATIONS
   ============================================================ */

// ---- 1. ANIMATED PROGRESS INDICATOR ----
function injectAnimatedProgressIndicator() {
  // Watch for spinner and enhance it
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) {
          // Find text containing "Analyzing"
          if (node.textContent && node.textContent.includes('Analyzing')) {
            enhanceAnalyzingSpinner(node);
          }
          // Also check children
          const spinnerText = node.querySelector ? node.querySelector('[data-testid="stSpinner"], .stSpinner') : null;
          if (spinnerText) {
            enhanceAnalyzingSpinner(spinnerText.parentElement);
          }
        }
      });
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

function enhanceAnalyzingSpinner(container) {
  if (!container || container.dataset.progressEnhanced) return;
  container.dataset.progressEnhanced = 'true';

  const progressHTML = document.createElement('div');
  progressHTML.className = 'enhanced-progress-container';
  progressHTML.innerHTML = `
    <div class="progress-indicator" style="
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem;
      background: linear-gradient(145deg, #0a0a0a, #000000);
      border-radius: 24px;
      border: 1px solid rgba(255,255,255,0.12);
      margin: 1rem 0;
    ">
      <div class="progress-ring" style="
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        animation: pulseRing 2s infinite;
        box-shadow: 0 0 30px rgba(0, 113, 227, 0.3);
      ">
        <div style="font-size: 2rem;">🌱</div>
        <div style="
          position: absolute;
          width: 100%;
          height: 100%;
          border-radius: 50%;
          border: 3px solid transparent;
          border-top-color: #0071e3;
          border-right-color: #34c759;
          animation: spinnerRotate 1s linear infinite;
        "></div>
      </div>
      
      <div style="margin-top: 1.5rem; text-align: center;">
        <div style="color: white; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">Analyzing Root Structure</div>
        <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">Processing image data...</div>
      </div>
      
      <div class="step-timeline" style="
        display: flex;
        justify-content: space-between;
        width: 100%;
        max-width: 300px;
        margin-top: 1.5rem;
        position: relative;
      ">
        <div class="step-item active" style="display: flex; flex-direction: column; align-items: center; z-index: 1;">
          <div style="width: 30px; height: 30px; border-radius: 50%; background: #0071e3; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8rem; box-shadow: 0 0 15px rgba(0,113,227,0.5);">1</div>
          <div style="font-size: 0.7rem; color: white; margin-top: 0.5rem;">Upload</div>
        </div>
        <div class="step-item active" style="display: flex; flex-direction: column; align-items: center; z-index: 1;">
          <div style="width: 30px; height: 30px; border-radius: 50%; background: #0071e3; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8rem; animation: pulseRing 1.5s infinite;">2</div>
          <div style="font-size: 0.7rem; color: white; margin-top: 0.5rem;">Analyze</div>
        </div>
        <div class="step-item" style="display: flex; flex-direction: column; align-items: center; z-index: 1;">
          <div style="width: 30px; height: 30px; border-radius: 50%; background: #1a1a1a; border: 2px solid rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.5); font-size: 0.8rem;">3</div>
          <div style="font-size: 0.7rem; color: rgba(255,255,255,0.5); margin-top: 0.5rem;">Results</div>
        </div>
      </div>
      
      <div class="progress-bar-container" style="
        width: 100%;
        max-width: 300px;
        height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
        overflow: hidden;
        margin-top: 1.5rem;
      ">
        <div class="progress-bar-fill" style="
          height: 100%;
          background: linear-gradient(90deg, #0071e3, #34c759);
          border-radius: 3px;
          width: 60%;
          animation: progressPulse 1.5s ease-in-out infinite;
        "></div>
      </div>
    </div>
  `;

  container.insertBefore(progressHTML, container.firstChild);
}

// ---- 2. DARK MODE TOGGLE ----
function injectDarkModeToggle() {
  if (document.querySelector('.theme-toggle-container')) return;

  const toggleContainer = document.createElement('div');
  toggleContainer.className = 'theme-toggle-container';
  toggleContainer.innerHTML = `
    <div class="theme-toggle" id="themeToggle" style="
      position: fixed;
      top: 20px;
      right: 20px;
      width: 50px;
      height: 50px;
      border-radius: 50%;
      background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
      border: 1px solid rgba(255, 255, 255, 0.1);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.5rem;
      z-index: 9999;
      transition: all 0.3s ease;
      box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    ">
      🌙
    </div>
    <div class="theme-menu" id="themeMenu" style="
      position: fixed;
      top: 80px;
      right: 20px;
      background: linear-gradient(145deg, #0a0a0a, #000000);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 16px;
      padding: 0.5rem;
      z-index: 9998;
      display: none;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    ">
      <button class="theme-option" data-theme="dark" style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        padding: 0.75rem 1rem;
        background: transparent;
        border: none;
        color: white;
        font-size: 0.9rem;
        cursor: pointer;
        border-radius: 10px;
        transition: background 0.2s ease;
      ">
        <span>🌙</span> Dark Mode
      </button>
      <button class="theme-option" data-theme="light" style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        padding: 0.75rem 1rem;
        background: transparent;
        border: none;
        color: white;
        font-size: 0.9rem;
        cursor: pointer;
        border-radius: 10px;
        transition: background 0.2s ease;
      ">
        <span>☀️</span> Light Mode
      </button>
      <button class="theme-option" data-theme="auto" style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        padding: 0.75rem 1rem;
        background: transparent;
        border: none;
        color: white;
        font-size: 0.9rem;
        cursor: pointer;
        border-radius: 10px;
        transition: background 0.2s ease;
      ">
        <span>🔄</span> Auto
      </button>
    </div>
  `;

  document.body.appendChild(toggleContainer);

  // Event listeners
  const toggle = document.getElementById('themeToggle');
  const menu = document.getElementById('themeMenu');

  toggle.addEventListener('click', () => {
    menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
    toggle.style.transform = menu.style.display === 'block' ? 'scale(1.1) rotate(15deg)' : '';
  });

  toggle.addEventListener('mouseenter', () => {
    toggle.style.transform = 'scale(1.1) rotate(15deg)';
    toggle.style.boxShadow = '0 8px 30px rgba(0, 0, 0, 0.5)';
  });

  toggle.addEventListener('mouseleave', () => {
    if (menu.style.display === 'none') {
      toggle.style.transform = '';
    }
    toggle.style.boxShadow = '0 5px 20px rgba(0, 0, 0, 0.3)';
  });

  document.querySelectorAll('.theme-option').forEach(btn => {
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(255, 255, 255, 0.1)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'transparent';
    });
    btn.addEventListener('click', () => {
      const theme = btn.dataset.theme;
      applyThemeMode(theme);
      menu.style.display = 'none';
      toggle.innerHTML = theme === 'light' ? '☀️' : theme === 'auto' ? '🔄' : '🌙';
      showEnhancedToast('Theme Changed', `Switched to ${theme} mode`, 'info', 2000);
    });
  });

  // Close menu on outside click
  document.addEventListener('click', (e) => {
    if (!toggleContainer.contains(e.target)) {
      menu.style.display = 'none';
    }
  });
}

function applyThemeMode(mode) {
  localStorage.setItem('smartroot-theme', mode);

  if (mode === 'auto') {
    mode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  document.documentElement.setAttribute('data-theme', mode);

  // For now, we keep dark mode as default since the entire UI is designed for it
  // Light mode would require significant CSS changes
}

// ---- 3. GLASSMORPHISM EFFECTS ----
function applyGlassmorphism() {
  // Apply to result boxes and cards
  const glassElements = document.querySelectorAll('.stress-result-box, .analytics-box, .section-card');

  glassElements.forEach(el => {
    if (el.dataset.glassApplied) return;
    el.dataset.glassApplied = 'true';

    el.style.backdropFilter = 'blur(10px)';
    el.style.webkitBackdropFilter = 'blur(10px)';
  });

  // Apply glass effect to sidebar
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  if (sidebar && !sidebar.dataset.glassApplied) {
    sidebar.dataset.glassApplied = 'true';
    sidebar.style.backdropFilter = 'blur(20px)';
    sidebar.style.webkitBackdropFilter = 'blur(20px)';
    sidebar.style.background = 'rgba(10, 10, 10, 0.9)';
  }
}

// ---- 4. MICRO-ANIMATIONS ----
function applyMicroAnimations() {
  // Animate icons on hover
  document.querySelectorAll('.feature-icon, .analytics-icon, .result-card-icon').forEach(icon => {
    if (icon.dataset.animationApplied) return;
    icon.dataset.animationApplied = 'true';

    icon.addEventListener('mouseenter', () => {
      icon.style.animation = 'iconGrow 0.5s ease';
    });

    icon.addEventListener('animationend', () => {
      icon.style.animation = '';
    });
  });

  // Download button bounce
  document.querySelectorAll('[data-testid="stDownloadButton"] button, .stDownloadButton button').forEach(btn => {
    if (btn.dataset.animationApplied) return;
    btn.dataset.animationApplied = 'true';

    btn.addEventListener('mouseenter', () => {
      const svg = btn.querySelector('svg');
      if (svg) {
        svg.style.animation = 'iconBounce 0.6s ease infinite';
      }
    });

    btn.addEventListener('mouseleave', () => {
      const svg = btn.querySelector('svg');
      if (svg) {
        svg.style.animation = '';
      }
    });
  });
}

// ---- CONFETTI EFFECT ----
function triggerConfetti() {
  const colors = ['#34c759', '#0071e3', '#ff9500', '#af52de', '#ff3b30'];
  const confettiCount = 50;

  for (let i = 0; i < confettiCount; i++) {
    const confetti = document.createElement('div');
    confetti.className = 'confetti-piece';
    confetti.style.cssText = `
      position: fixed;
      width: ${Math.random() * 10 + 5}px;
      height: ${Math.random() * 10 + 5}px;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      left: ${Math.random() * 100}vw;
      top: -20px;
      border-radius: ${Math.random() > 0.5 ? '50%' : '0'};
      animation: confettiFall ${Math.random() * 2 + 2}s ease-out forwards;
      animation-delay: ${Math.random() * 0.5}s;
      z-index: 99999;
      pointer-events: none;
    `;

    document.body.appendChild(confetti);

    setTimeout(() => confetti.remove(), 4000);
  }
}

// ---- SUCCESS PULSE EFFECT ----
function triggerSuccessPulse(element) {
  if (!element) return;

  element.style.animation = 'successPulse 1.5s ease-out';
  setTimeout(() => {
    element.style.animation = '';
  }, 1500);
}

// ---- 5. GRADIENT ACCENT COLORS ----
function applyGradientAccents() {
  // Apply gradient to health-related values
  document.querySelectorAll('[data-testid="stMetricValue"]').forEach(value => {
    if (value.dataset.gradientApplied) return;
    value.dataset.gradientApplied = 'true';

    const text = value.textContent || '';
    const numMatch = text.match(/(\d+)/);

    if (numMatch) {
      const num = parseInt(numMatch[1]);

      if (num >= 80) {
        // Excellent - Green gradient
        value.style.background = 'linear-gradient(135deg, #34c759, #30d158)';
        value.style.webkitBackgroundClip = 'text';
        value.style.webkitTextFillColor = 'transparent';
        value.style.backgroundClip = 'text';
      } else if (num >= 60) {
        // Good - Blue gradient
        value.style.background = 'linear-gradient(135deg, #0071e3, #5ac8fa)';
        value.style.webkitBackgroundClip = 'text';
        value.style.webkitTextFillColor = 'transparent';
        value.style.backgroundClip = 'text';
      } else if (num >= 40) {
        // Warning - Orange gradient
        value.style.background = 'linear-gradient(135deg, #ff9500, #ffcc00)';
        value.style.webkitBackgroundClip = 'text';
        value.style.webkitTextFillColor = 'transparent';
        value.style.backgroundClip = 'text';
      } else {
        // Critical - Red gradient
        value.style.background = 'linear-gradient(135deg, #ff3b30, #ff6961)';
        value.style.webkitBackgroundClip = 'text';
        value.style.webkitTextFillColor = 'transparent';
        value.style.backgroundClip = 'text';
      }
    }
  });

  // Apply animated gradient border to focused inputs
  document.querySelectorAll('input, select, textarea').forEach(input => {
    if (input.dataset.gradientBorderApplied) return;
    input.dataset.gradientBorderApplied = 'true';

    input.addEventListener('focus', () => {
      input.style.outline = 'none';
      input.style.boxShadow = '0 0 0 2px #000000, 0 0 0 4px #0071e3';
    });

    input.addEventListener('blur', () => {
      input.style.boxShadow = '';
    });
  });
}

// ---- CHECK FOR EXCELLENT HEALTH SCORE ----
function checkForExcellentScore() {
  const healthValues = document.querySelectorAll('[data-testid="stMetricValue"]');

  healthValues.forEach(value => {
    const text = value.textContent || '';
    if (text.includes('/100')) {
      const numMatch = text.match(/(\d+)/);
      if (numMatch && parseInt(numMatch[1]) >= 90 && !value.dataset.confettiTriggered) {
        value.dataset.confettiTriggered = 'true';
        // Trigger confetti for excellent score
        setTimeout(() => {
          triggerConfetti();
          triggerSuccessPulse(value.closest('[data-testid="metric-container"]'));
          showEnhancedToast('🎉 Excellent!', 'Root health score is outstanding!', 'success', 4000);
        }, 500);
      }
    }
  });
}

// ---- MASTER INITIALIZATION ----
function initializeAllVisualEnhancements() {
  setTimeout(() => {
    // Core UI injections
    injectQuickActionsToolbar();
    injectToastContainer();
    injectDarkModeToggle();

    // Visual enhancements
    injectAnimatedProgressIndicator();
    applyGlassmorphism();
    applyMicroAnimations();
    applyGradientAccents();

    // Tooltips and cards
    injectTooltipsOnLabels();
    enhanceResultCards();
    enhanceDragAndDrop();
    addStaggerAnimations();

    // Check for excellent scores
    checkForExcellentScore();

    // Welcome toast
    showEnhancedToast('SmartRoot AI', 'Ready to analyze your plant roots!', 'success', 4000);
  }, 1000);

  // Continuous observer for dynamic content
  const observer = new MutationObserver(() => {
    applyGlassmorphism();
    applyMicroAnimations();
    applyGradientAccents();
    injectTooltipsOnLabels();
    enhanceResultCards();
    enhanceDragAndDrop();
    addStaggerAnimations();
    checkForExcellentScore();
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

// ---- EXPORT ALL ----
window.SmartRootUI = {
  showEnhancedToast,
  triggerConfetti,
  triggerSuccessPulse,
  injectQuickActionsToolbar,
  injectDarkModeToggle,
  applyGlassmorphism,
  applyMicroAnimations,
  applyGradientAccents,
  initializeAllVisualEnhancements
};

// ---- AUTO-INITIALIZE ----
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeAllVisualEnhancements);
} else {
  initializeAllVisualEnhancements();
}
