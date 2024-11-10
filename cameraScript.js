// Refresh the live feed every 1000ms (1 second)
const img = document.getElementById('liveFeed');
setInterval(() => {
    img.src = `http://192.168.50.9:5001/video_feed?timestamp=${new Date().getTime()}`;
}, 1000);

// Example function to activate modes - replace with actual functionality
function activateMode(mode) {
    // Implement mode activation logic here
    alert(`Mode ${mode} activated!`);
}