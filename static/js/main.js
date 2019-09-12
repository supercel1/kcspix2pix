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
        }

        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = width;
        ctx.strokeStyle = color;

        if (lastPosition.x == null || lastPosition.y == null) {
            ctx.moveTo(x, y);
        } else {
            ctx.moveTo(lastPosition.x, lastPosition.y);
        }

        ctx.lineTo(x, y);
        ctx.stroke();

        lastPosition.x = x;
        lastPosition.y = y;
    }

    function clear() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function dragStart(event) {
        ctx.beginPath();
        isDrag = true;
    }

    function dragEnd(event) {
        ctx.closePath();
        isDrag = false;

        lastPosition.x = null;
        lastPosition.y = null;
    }

    function chanegeColor() {
        const color = document.getElementById('color').value;
        console.log(color);
        ctx.strokeStyle = color;
    }

    function changeWidth() {
        let width = document.getElementById('width');
        ctx.lineWidth = width;
    }

    function changePencil() {
        ctx.globalCompositeOperation = 'source-over';
    }

    function changeEraser() {
        ctx.globalCompositeOperation = 'destination-out';
    }

    function canvasDraw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let img = new Image();
        img.src = uploadImgSrc;
        img.onload = () => {
            let w, h;
            const maxSize = 256;
            if (img.width > img.height) {
                w = maxSize;
                h = Math.floor(maxSize * img.height / img.width);
            } else {
                w = Math.floor(maxSize * img.width / img.height);
                h = maxSize;
            }
            const start = [(maxSize - w) / 2, (maxSize - h) / 2];
            ctx.fillRect(0, 0, 286, 286);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, start[0], start[1], w, h);
        }
    }

    $('input[type=file]').change(() => {
        const file = $('input[type=file]').prop('files')[0];

        let fd = new FormData();
        fd.append($('input[type=file]').attr('name'), file);

        $.ajax({
            url: 'file',
            type: 'POST',
            data: fd,
            processData: false,
            contentType: false,
            enctype: 'multipart/form-data'
        })
            .done((data) => {
                console.log(data);
                const image = document.getElementById('output');
                image.src = 'http://localhost:8000/' + data['file_path'];
                console.log(data['file_path']);
            })
            .fail((textStatus) => {
                console.log(textStatus);
            });

        const reader = new FileReader();
        reader.onload = () => {
            uploadImgSrc = reader.result;
            canvasDraw();
        }
        reader.readAsDataURL(file);
    });

    const btn = $('#predict');

    btn.on('click', () => {
        btn.prop('disabled', true);
        btn.html('推測中');

        let dataURI = canvas.toDataURL('image/png');
        dataURI = dataURI.replace(/^data:image\/png;base64,/, '');

        let postData = dataURI; 
        $.ajax({
            url: 'image',
            type: 'POST',
            data: postData,
            contentType: false,
            processData: false,
            dataType: 'json',
        })
            .done((data, textStatus, jqxHR) => {
                console.log(data);
                let dataURI = data['fake_image_path'];
                const img = document.getElementById('output');

                img.src = 'http://localhost:8000/' +  dataURI;
            })
            .fail((jqXHR, textStatus, errorThrown) => {
                console.log(jqXHR);
                console.log(textStatus);
                console.log(errorThrown);
            })
            .always(() => {
                btn.prop('disabled', false);
                btn.html('予測する');
            });
    });

    // マウス操作やボタンクリック時のイベント処理を定義する
    function initEventHandler() {
        const clearButton = document.querySelector('#clear-button');
        const colorButton = document.querySelector('#color');
        const widthButton = document.querySelector('#width');
        const pencilButton = document.querySelector('#pencil');
        const eraserButton = document.querySelector('#eraser');

        clearButton.addEventListener('click', clear);
        colorButton.addEventListener('click', chanegeColor);
        widthButton.addEventListener('click', changeWidth);
        pencilButton.addEventListener('click', changePencil);
        eraserButton.addEventListener('click', changeEraser);
        canvas.addEventListener('mousedown', dragStart);
        canvas.addEventListener('mouseup', dragEnd);
        canvas.addEventListener('mouseout', dragEnd);
        canvas.addEventListener('mousemove', (event) => {
            // console.log(event);

            draw(event.layerX, event.layerY);
        });
    }

    initEventHandler();
});