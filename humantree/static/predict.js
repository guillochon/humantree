var width = 512;
var height = 512;
var img = undefined;
var mets = undefined;
var address = undefined;

async function loadModel() {
  // Load model.
  // document.getElementById('loader').style.display = "block"
  console.log('Loading model...');
  this.model = await tf.loadModel('assets/model.json');
  console.log('Model loaded.');
  document.getElementById('loader').style.display = "none";
  document.getElementById('btn').disabled = false;
}

function predict(imageData) {

  return tf.tidy(() => {

    // Convert the canvas pixels to a Tensor of the matching shape
    img = tf.tensor(imageData);
    img = tf.cast(img, 'float32');
    mean = tf.scalar(102.32204);
    std = tf.scalar(50.741535);
    img = img.sub(mean).div(std);
    img = img.reshape([1, 512, 512, 3]);
    img = this.model.predict(img).mul(
      tf.scalar(255.0)).dataSync();

    const output = Uint8ClampedArray.from(img);

    drawImage(output);

    return output;
  });
}

function drawImage(image) {
  var buffer = new Uint8ClampedArray(width * height * 4);

  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      var ipos = (y * width + x);
      var pos = ipos * 4; // position in buffer based on x and y
      buffer[pos] = image[ipos]; // some R value [0, 255]
      buffer[pos + 1] = image[ipos]; // some G value
      buffer[pos + 2] = image[ipos]; // some B value
      buffer[pos + 3] = 255; // set alpha channel
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

function metrics(image) {
  $.ajax({
    url: '/metrics',
    method: 'POST',
    dataType: 'json',
    data: {'image': image.join(','), 'address': address},
    success: function (data) {
      mets = data;
      document.getElementById('metrizer').style.display = "none"
      var met_str = '';
      met_str += '<strong>' + String((
        mets['fraction'] * 100).toFixed(1)) + '%</strong> of property shaded.<br><br>'
      met_str += 'Heating & cooling costs: $' + String((
        mets['cost']).toFixed(0)) + ' per year.<br>'
      met_str += 'Savings from trees: <strong>$' + String((
        mets['savings']).toFixed(0)) + ' per year</strong>.<br><br>'
      met_str += 'Noise abatement from trees: <strong>' + String((
        mets['noise_abatement']).toFixed(1)) + ' dB</strong>.<br><br>'
      if (mets['house_value'] > 0.0) {
        met_str += 'Estimated house value: $' + String((
          mets['house_value']).toFixed(0)).replace(
            /\B(?=(\d{3})+(?!\d))/g, ",") + '.<br>'
        met_str += 'Value of trees: <strong>$' + String((
          mets['value_increase']).toFixed(0)).replace(
            /\B(?=(\d{3})+(?!\d))/g, ",") + '</strong>.'
      }
      document.getElementById('metrics').innerHTML = met_str;
      document.getElementById('metrics').style.display = "block"
    }
  })
}

function handleProcess(response) {
  var sat = $("#satellite");
  sat.attr("src", "queries/" + response);
  sat.width(width / 2);
  sat.height(height / 2);
  $.ajax({
    url: "queries/" + response.replace('.png', '.json'),
    dataType: 'json',
    success: function (json) {
      img = predict(json);
      document.getElementById('analyzer').style.display = "none"
      document.getElementById('images').style.display = "block"
    },
    complete: function () {
      document.getElementById('metrizer').style.display = "block"
      metrics(img);
    }
  });
  // img = $.getJSON("queries/" + response.replace(
  //   '.png', '.json'), function(json) {
  //   img = predict(json);
  //   document.getElementById('analyzer').style.display = "none"
  //   document.getElementById('images').style.display = "block"
  //   return img;
  // });
}

$(function() {
  $('#process').submit(function(e) {
    e.preventDefault(); // prevent the form from 'submitting'
    address = document.getElementById('address').value
    document.getElementById('analyzer').style.display = "block"
    document.getElementById('images').style.display = "none"
    document.getElementById('metrics').style.display = "none"

    var url = e.target.action; // get the target
    var formData = $(this).serialize(); // get form data
    $.post(url, formData, handleProcess);
  })
})

loadModel();
