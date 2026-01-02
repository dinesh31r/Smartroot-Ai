// static/script.js - Light Theme Enhancements
document.addEventListener('DOMContentLoaded', function() {
    // Add subtle hover effects to cards
    const cards = document.querySelectorAll('[data-testid="stMetric"], .element-container');
    cards.forEach(card => {
        card.style.transition = 'all 0.3s ease';
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-4px)';
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });

    // Add loading animation to buttons
    const buttons = document.querySelectorAll('.stButton button');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Add ripple effect
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size/2;
            const y = e.clientY - rect.top - size/2;
            
            ripple.style.cssText = `
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: scale(0);
                animation: ripple 0.6s linear;
                width: ${size}px;
                height: ${size}px;
                top: ${y}px;
                left: ${x}px;
            `;
            
            this.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        });
    });

    // Add to stylesheet
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // Update file uploader text dynamically
    const uploader = document.querySelector('[data-testid="stFileUploader"]');
    if (uploader) {
        const dropText = uploader.querySelector('p');
        if (dropText) {
            dropText.innerHTML = '🌿 <strong>Drag & Drop</strong> or <strong>Click to Browse</strong>';
            dropText.style.color = '#64748b';
            dropText.style.fontWeight = '500';
        }
    }

    // Add live timestamp
    const timestamp = document.createElement('div');
    timestamp.className = 'timestamp';
    timestamp.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-size: 0.8rem;
        color: #94a3b8;
        background: rgba(255, 255, 255, 0.8);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.6);
        z-index: 1000;
    `;
    
    function updateTime() {
        const now = new Date();
        timestamp.textContent = `🕒 ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
    }
    
    updateTime();
    setInterval(updateTime, 60000);
    document.body.appendChild(timestamp);
});