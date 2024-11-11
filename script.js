function toggleMenu() {
    const sideMenu = document.getElementById('sideMenu');
    if (sideMenu.classList.contains('open')) {
        sideMenu.classList.remove('open');
    } else {
        sideMenu.classList.add('open');
    }
}

function showLogin() {
    window.location.href = 'login.html'; // Redirect to login page
}

function goToHome() {
    document.getElementById('cameraFeed').style.display = 'none';
    document.getElementById('homeScreen').style.display = 'flex';
}

function goToCameraFeed() {
    document.getElementById('homeScreen').style.display = 'none';
    document.getElementById('cameraFeed').style.display = 'flex';

    setInterval(() => {
        const img = document.getElementById('liveFeed');
        img.src = `https://5f2ff07snubp.share.zrok.io/video_feed?timestamp=${new Date().getTime()}`;
    }, 1000);
}

function activateMode(mode) {
    fetch(`https://activationmode.share.zrok.io/activate_mode/${mode}`, { 
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            alert(data.status);
        } else {
            alert('Failed to activate mode: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to connect to Raspberry Pi');
    });
}
