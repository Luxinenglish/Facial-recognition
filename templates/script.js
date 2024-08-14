document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');

    // Accéder à la caméra
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(error => {
            console.error('Erreur d\'accès à la caméra:', error);
        });
});
