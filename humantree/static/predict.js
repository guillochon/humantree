var width = 512;
var height = 512;

async function loadModel() {
  // Load model.
  console.log('Loading model...');
  this.model = await tf.loadModel('assets/model.json');
  console.log('Model loaded.');
}

async function predict(imageData) {

  await tf.tidy(() => {

    // Convert the canvas pixels to a Tensor of the matching shape
    img = tf.tensor(imageData);
    img = tf.cast(img, 'float32');
    mean = tf.scalar(102.32204);
    std = tf.scalar(50.741535);
    img = img.sub(mean).div(std);
    img = img.reshape([1, 512, 512, 3]);

    const output = Uint8ClampedArray.from(this.model.predict(img).mul(
      tf.scalar(255.0)).dataSync());

    drawImage(output);
  });

}

function drawImage(image) {
  var buffer = new Uint8ClampedArray(width * height * 4);

  for(var y = 0; y < height; y++) {
    for(var x = 0; x < width; x++) {
        var ipos = (y * width + x);
        var pos = ipos * 4; // position in buffer based on x and y
        buffer[pos  ] = image[ipos];   // some R value [0, 255]
        buffer[pos+1] = image[ipos];   // some G value
        buffer[pos+2] = image[ipos];   // some B value
        buffer[pos+3] = 255;           // set alpha channel
    }
  }

  // create off-screen canvas element
  var canvas = document.createElement('canvas'),
      ctx = canvas.getContext('2d');

  canvas.width = width;
  canvas.height = height;

  // create imageData object
  var idata = ctx.createImageData(width, height);

  // set our buffer as source
  idata.data.set(buffer);

  // update canvas with new data
  ctx.putImageData(idata, 0, 0);

  var dataUri = canvas.toDataURL(); // produces a PNG file

  var cimage = new Image();
  cimage.id = 'predict';
  cimage.src = dataUri;
  var el = document.getElementById('predict');
  el.parentNode.replaceChild(cimage, el);
  var pred = $("#predict");
  pred.width(width / 2);
  pred.height(height / 2);
}

$(function () {
    $('#process').submit(function (e) {
        e.preventDefault();  // prevent the form from 'submitting'
        document.getElementById('loader').style.display = "block"
        document.getElementById('images').style.display = "none"

        var url = e.target.action;  // get the target
        var formData = $(this).serialize(); // get form data
        $.post(url, formData, function (response) { // send; response.data will be what is returned
            var sat = $("#satellite");
            sat.attr("src", "queries/" + response);
            sat.width(width / 2);
            sat.height(height / 2);
            $.getJSON("queries/" + response.replace(
                '.png', '.json'), function(json) {
              predict(json);
              document.getElementById('loader').style.display = "none"
              document.getElementById('images').style.display = "block"
            });
        });
    })
})

loadModel();
