window.addEventListener('load', () => {
    const canvas = document.querySelector('#canvas');
    const ctx = canvas.getContext('2d');

    const lastPosition = { x: null, y: null };


    let isDrag = false;

    function draw(x, y) {

        const color = document.getElementById('color').value;
        const width = document.getElementById('width').value;

        if (!isDrag) {
            return;
        };

        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = width;
        ctx.strokeStyle = color;

        if (lastPosition.x == null || lastPosition.y == null) {
            ctx.moveTo(x, y);
        } else {
            ctx.moveTo(lastPosition.x, lastPosition.y);
        };

        ctx.lineTo(x, y);
        ctx.stroke();

        lastPosition.x = x;
        lastPosition.y = y;
    };

    function clear() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    };

    function dragStart(event) {
        ctx.beginPath();
        isDrag = true;
    };

    function dragEnd(event) {
        ctx.closePath();
        isDrag = false;

        lastPosition.x = null;
        lastPosition.y = null;
    };

    function chanegeColor() {
        const color = document.getElementById('color').value;
        console.log(color)
        ctx.strokeStyle = color;
    };

    function changeWidth() {
        const width = document.getElementById('width');
        ctx.lineWidth = width;
    };

    function changePencil() {
        ctx.globalCompositeOperation = 'source-over';
    };

    function changeEraser() {
        ctx.globalCompositeOperation = 'destination-out';
    };

    function imageSave() {
        const dataURI = canvas.toDataURL('image/jpeg');
        const image = document.getElementById('output');
        image.src = dataURI;
    };

    // マウス操作やボタンクリック時のイベント処理を定義する
    function initEventHandler() {
        const clearButton = document.querySelector('#clear-button');
        const colorButton = document.querySelector('#color');
        const widthButton = document.querySelector('#width');
        const pencilButton = document.querySelector('#pencil');
        const eraserButton = document.querySelector('#eraser');
        const saveButton = document.querySelector('#save');

        clearButton.addEventListener('click', clear);
        colorButton.addEventListener('click', chanegeColor);
        widthButton.addEventListener('click', changeWidth);
        pencilButton.addEventListener('click', changePencil);
        eraserButton.addEventListener('click', changeEraser);
        saveButton.addEventListener('click', imageSave);
        canvas.addEventListener('mousedown', dragStart);
        canvas.addEventListener('mouseup', dragEnd);
        canvas.addEventListener('mouseout', dragEnd);
        canvas.addEventListener('mousemove', (event) => {
            // console.log(event);

            draw(event.layerX, event.layerY);
        });
    };

    initEventHandler();
});