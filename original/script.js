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
        img.src = `https://camerastream1.share.zrok.io/video_feed?timestamp=${new Date().getTime()}`;
    }, 1000);
}



function sendKey(key) {
    fetch("https://activationmode1.share.zrok.io/key_input", {  // Use your server URL here
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ key: key })
    })
    .then(response => response.json())
    .then(data => console.log(`Key ${key} sent: `, data.status))
    .catch(error => console.error("Error sending key:", error));
}
