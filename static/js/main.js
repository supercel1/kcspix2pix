window.addEventListener('load', () => {
    const canvas = document.querySelector('#canvas');
    const ctx = canvas.getContext('2d');

    const lastPosition = { x: null, y: null };


    let isDrag = false;

    function draw(x, y) {

        if (!isDrag) {
            return;
        }

        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = 5;
        ctx.strokeStyle = 'black';

        if (lastPosition.x == null || lastPosition.y == null) {
            ctx.moveTo(x, y);
        } else {
            ctx.moveTo(lastPosition.x, lastPosition.y);
        }

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

    // マウス操作やボタンクリック時のイベント処理を定義する
    function initEventHandler() {
        const clearButton = document.querySelector('#clear-button');
        clearButton.addEventListener('click', clear);
        canvas.addEventListener('mousedown', dragStart);
        canvas.addEventListener('mouseup', dragEnd);
        canvas.addEventListener('mouseout', dragEnd);
        canvas.addEventListener('mousemove', (event) => {
            console.log(event);

            draw(event.layerX, event.layerY);
        });
    };

    initEventHandler();
});