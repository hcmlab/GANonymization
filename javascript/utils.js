function clearCanvas(ctx) {
    const canvas = ctx.canvas;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fill();
}

const delay = ms => new Promise(res => setTimeout(res, ms));