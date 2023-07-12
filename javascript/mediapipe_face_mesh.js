function setup_face_mesh(canvas, image_size, on_updated_callback) {
    var ctx = canvas.getContext('2d');
    const config = {
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@${VERSION}/${file}`;
        }
    };
    const solutionOptions = {
        selfieMode: true,
        enableFaceGeometry: false,
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    };
    const faceMesh = new FaceMesh(config);
    faceMesh.setOptions(solutionOptions);
    faceMesh.onResults(results => {
        if (results.multiFaceLandmarks.length > 0) {
            clearCanvas(ctx);
            ctx.save();
            var x_min = 1;
            var x_max = 0;
            var y_min = 1;
            var y_max = 0;
            for (const landmark of results.multiFaceLandmarks[0]) {
                if (landmark.x > x_max) {
                    x_max = landmark.x;
                } else if (landmark.x < x_min) {
                    x_min = landmark.x;
                }
                if (landmark.y > y_max) {
                    y_max = landmark.y;
                } else if (landmark.y < y_min) {
                    y_min = landmark.y;
                }
            }
            var aspect_ratio = (y_max - y_min) / (x_max - x_min);
            const face_height = image_size;
            const face_width = image_size / aspect_ratio;
            for (const landmark of results.multiFaceLandmarks[0]) {
                const x_normalized = (landmark.x - x_min) / (x_max - x_min);
                const y_normalized = (landmark.y - y_min) / (y_max - y_min);
                const x_center = canvas.width / 2;
                const y_center = canvas.height / 2;
                const x = x_normalized * face_width + x_center - face_width / 2;
                const y = y_normalized * face_height + y_center - face_height / 2;
                // Decrease the size of the arc to compensate for the scale()
                const circle = new Path2D();
                circle.arc(x, y, 1, 0, 2 * Math.PI);
                ctx.fillStyle = 'white';
                ctx.fill(circle);
            }
            ctx.restore();

            on_updated_callback();
        }
    });
    return faceMesh;
}