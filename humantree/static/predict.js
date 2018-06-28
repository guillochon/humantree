var width = 512;
var height = 512;
var img = undefined;
var mets = undefined;
var address = undefined;

async function loadModel() {
  console.log('Loading model...');
  this.model = await tf.loadModel('assets/model.json');
  console.log('Model loaded.');
  document.getElementById('loader').style.display = 'none';
  document.getElementById('btn').disabled = false;
}

function predict(imageData, propRadius) {

  console.log(tf.memory().numBytes);
  console.log(tf.getBackend());
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

    drawImage(output, propRadius);

    return output;
  });
}

function drawImage(image, propRadius) {
  var buffer = new Uint8ClampedArray(width * height * 4);

  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      var ipos = (y * width + x);
      var pos = ipos * 4; // position in buffer based on x and y
      var bit = image[ipos] > 127 ? 0 : 255;
      var alp = image[ipos] > 127 ? 0 : 100;
      buffer[pos] = bit; // some R value [0, 255]
      buffer[pos + 1] = 0; // some G value
      buffer[pos + 2] = bit; // some B value
      buffer[pos + 3] = alp; // set alpha channel
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

  ctx.beginPath();
  // WARNING: 512/70 eyeballed, need exact calc.
  ctx.arc(width / 2, height / 2, 512.0 / 70.0 * propRadius,
    0.0, 2.0 * Math.PI);
  ctx.lineWidth = 5;
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = 'yellow';
  ctx.stroke();

  var dataUri = canvas.toDataURL(); // produces a PNG file

  var cimage = new Image();
  cimage.id = 'predict';
  cimage.src = dataUri;
  cimage.classList.add('validate-mask');
  var el = document.getElementById('predict');
  el.parentNode.replaceChild(cimage, el);
  var pred = $('#predict');
  pred.width(400);
  pred.height(400);
}

function metrics(image, head) {
  $.ajax({
    url: '/metrics',
    method: 'POST',
    dataType: 'json',
    data: {
      'image': image.join(','),
      'address': address,
      'head': head
    },
    success: function(data) {
      var savingsKnob = pureknob.createKnob(300, 250, 0.7);
      savingsKnob.setProperty('angleStart', -0.4 * Math.PI);
      savingsKnob.setProperty('angleEnd', 0.4 * Math.PI);
      savingsKnob.setProperty('colorFG', 'forestgreen');
      savingsKnob.setProperty('trackWidth', 0.4);
      savingsKnob.setProperty('valMin', 0);
      savingsKnob.setProperty('readonly', true);
      savingsKnob.setProperty('needle', true);
      savingsKnob.setProperty('colorEX', 'lightgreen');
      savingsKnob.setProperty('prefix', '$');
      savingsKnob.setProperty('showRange', true);

      var noiseKnob = pureknob.createKnob(300, 250, 0.7);
      noiseKnob.setProperty('angleStart', -0.4 * Math.PI);
      noiseKnob.setProperty('angleEnd', 0.4 * Math.PI);
      noiseKnob.setProperty('colorFG', 'forestgreen');
      noiseKnob.setProperty('trackWidth', 0.4);
      noiseKnob.setProperty('valMin', 0);
      noiseKnob.setProperty('readonly', true);
      noiseKnob.setProperty('needle', true);
      noiseKnob.setProperty('colorEX', 'lightgreen');
      noiseKnob.setProperty('suffix', 'dB');
      noiseKnob.setProperty('showRange', true);

      var valueKnob = pureknob.createKnob(300, 250, 0.7);
      valueKnob.setProperty('angleStart', -0.4 * Math.PI);
      valueKnob.setProperty('angleEnd', 0.4 * Math.PI);
      valueKnob.setProperty('colorFG', 'forestgreen');
      valueKnob.setProperty('trackWidth', 0.4);
      valueKnob.setProperty('valMin', 0);
      valueKnob.setProperty('readonly', true);
      valueKnob.setProperty('needle', true);
      valueKnob.setProperty('colorEX', 'lightgreen');
      valueKnob.setProperty('prefix', '$');
      valueKnob.setProperty('suffix', 'k');
      valueKnob.setProperty('showRange', true);

      mets = data;
      document.getElementById('metrizer').style.display = 'none'
      var met_str = '';
      met_str += '<span class="shade-fraction"><strong>' + String((
        mets['fraction'] * 100).toFixed(1)) + '%</strong> of property shaded.</span><br>'
      met_str += '<div class="dial-container" id="savings_knob" width=150 height=100></div>'
      met_str += '<div class="stat-summary">'
      if (mets['cost'] > 0.0) {
        met_str += 'Heating & cooling costs: $' + String((
          mets['cost']).toFixed(0)).replace(
          /\B(?=(\d{3})+(?!\d))/g, ',') + ' per year.<br>'
      }
      if (mets['savings'] > 0.0) {
        met_str += 'Savings from current trees: $' + String((
          mets['savings']).toFixed(0)) + ' per year.<br>'
        met_str += 'Savings by adding a large tree: <strong>$' + String((
          mets['one_tree_savings']).toFixed(1)).replace(
          /\B(?=(\d{3})+(?!\d))/g, ',') + ' per year</strong>.<br clear=all><br>'
        savingsKnob.setProperty('valMax', parseFloat(mets['max_savings'].toFixed(0)));
        savingsKnob.setValue(parseFloat(mets['savings'].toFixed(0)));
        savingsKnob.setProperty('excess', mets['one_tree_savings']);
        savingsKnob.setProperty('excessLabel', '+1🌳');
      }

      met_str += '</div>'
      met_str += '<div class="dial-container" id="noise_knob" width=150 height=100></div>'
      met_str += '<div class="stat-summary">'
      met_str += 'Noise abatement from current trees: ' + String((
        mets['noise_abatement']).toFixed(1)) + ' dB.<br>'
      met_str += 'Additional abatement by adding a large tree: <strong>' + String((
        mets['one_tree_noise']).toFixed(1)).replace(
        /\B(?=(\d{3})+(?!\d))/g, ',') + ' dB</strong>.<br clear=all><br>'
      met_str += '</div>'
      noiseKnob.setProperty('valMax', parseFloat(mets['max_noise_abatement'].toFixed(0)));
      noiseKnob.setValue(parseFloat(mets['noise_abatement'].toFixed(0)));
      noiseKnob.setProperty('excess', mets['one_tree_noise']);
      noiseKnob.setProperty('excessLabel', '+1🌳');

      if (mets['house_value'] > 0.0) {
        met_str += '<div class="dial-container" id="value_knob" width=150 height=100></div>'
        met_str += '<div class="stat-summary">'
        met_str += 'Estimated house value: $' + String((
            mets['house_value']).toFixed(0)).replace(
            /\B(?=(\d{3})+(?!\d))/g, ",") +
          ' (via <a href="https://www.zillow.com"><img width=50px src="static/zillow.gif"></a>).<br>';
        met_str += 'Value increase from present trees: $' + String((
          mets['value_increase']).toFixed(0)).replace(
          /\B(?=(\d{3})+(?!\d))/g, ',') + '.<br>'
        met_str += 'Value increase by adding a large tree: <strong>$' + String((
          mets['one_tree_value']).toFixed(0)).replace(
          /\B(?=(\d{3})+(?!\d))/g, ',') + '</strong>.'
        met_str += '</div>'
        valueKnob.setProperty('valMax',
          parseFloat((mets['max_value_increase'] / 1000.0).toFixed(0)));
        valueKnob.setValue(
          parseFloat((mets['value_increase'] / 1000.0).toFixed(0)));
        valueKnob.setProperty('excess', mets['one_tree_value'] / 1000.0);
        valueKnob.setProperty('excessLabel', '+1🌳');
      }
      document.getElementById('metrics').innerHTML = met_str;
      if (mets['savings'] > 0.0) {
        document.getElementById('savings_knob').appendChild(savingsKnob.node());
      }
      document.getElementById('noise_knob').appendChild(noiseKnob.node());
      if (mets['house_value'] > 0.0) {
        document.getElementById('value_knob').appendChild(valueKnob.node());
      }
      document.getElementById('metrics').style.display = 'block'
    }
  })
}

function handleProcess(response) {
  console.log(response);
  if (Object.keys(response).length === 0) {
    document.getElementById('invalid-address').style.display = 'block';
    document.getElementById('btn').disabled = false;
    document.getElementById('analyzer').style.display = 'none';
    document.getElementById('metrizer').style.display = 'none';
    return;
  }

  var path = 'path' in response ? response['path'] : '';
  var radius = 'radius' in response ? response['radius'] : 0.0;

  var sat = $('#satellite');
  sat.attr('src', 'queries/' + path);
  // sat.width(width / 2);
  // sat.height(height / 2);
  var head = path.split('.')[0];
  $.ajax({
    url: 'queries/' + path.replace('.png', '.json'),
    dataType: 'json',
    success: function(json) {
      img = predict(json, radius);
      document.getElementById('analyzer').style.display = 'none'
      document.getElementById('imwrap').style.display = 'block'
      document.getElementById('maskswitch').style.display = 'block'
    },
    complete: function() {
      document.getElementById('metrizer').style.display = 'block'
      document.getElementById('btn').disabled = false;
      metrics(img, head);
    }
  });
}

$(function() {
  $('#process').submit(function(e) {
    e.preventDefault(); // prevent the form from 'submitting'
    address = document.getElementById('address').value;
    if ($.trim(address) == '') {
      document.getElementById('invalid-address').style.display = 'block';
      return;
    }

    document.getElementById('btn').disabled = true;;
    document.getElementById('invalid-address').style.display = 'none';
    document.getElementById('imwrap').style.display = 'none';
    document.getElementById('metrics').style.display = 'none';
    document.getElementById('maskswitch').style.display = 'none';
    document.getElementById('analyzer').style.display = 'block';

    document.getElementById('gmap-link').href = (
      'https://maps.google.com/?q=' + encodeURI(address));

    var url = e.target.action; // get the target
    var formData = $(this).serialize(); // get form data
    $.post(url, formData, handleProcess);
  })
})

function initialize() {
  var input = document.getElementById('address');
  new google.maps.places.Autocomplete(input);
}

function toggleMask() {
  var checked = document.getElementById('maskcheck').checked;
  if (checked) {
    document.getElementById('predict').style.opacity = 1.0;
  } else {
    document.getElementById('predict').style.opacity = 0.0;
  }
}

google.maps.event.addDomListener(window, 'load', initialize);

loadModel();
