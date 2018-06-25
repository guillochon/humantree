var load_time;

document.onkeyup = function(e) {
  if (e.which == 71) {
    document.getElementById('good').click();
  }
  else if (e.which == 79) {
    document.getElementById('okay').click();
  }
  else if (e.which == 66) {
    document.getElementById('bad').click();
  }
  else if (e.which == 83) {
    document.getElementById('maskcheck').click();
  }
};

function insertSrcs() {
  // Flip a coin to see if we validate queries or training set.
  var tids = document.head.querySelector(
    "[property=ids]").content.split(',');
  var qids_str = document.head.querySelector(
    "[property=qids]").content;
  var qids = qids_str.split(',');
  if ((Math.random() > 0.5) || (qids_str == '')) {
    var dir = 'parcels';
    var ids = tids;
  } else {
    var dir = 'queries';
    var ids = qids;
  }
  var iids = document.head.querySelector("[property=iids]").content;

  var i = Math.floor(Math.random() * ids.length);
  var idi = ids[i];
  document.getElementById('satellite').src = dir + '/' + idi + '.png';
  document.getElementById('mask').src = dir + '/' + idi + '-outline.png';
  document.getElementById('mask-num').value = idi;
  document.getElementById('iids').value = (
    iids === '' ? idi : (iids + ',' + idi));

  load_time = Date.now();
  var captcha = document.head.querySelector("[property=captcha]").content;
  if (captcha === "False") {
      setTimeout(
        function() {
          enableBtn();
      }, 1500);
  }
}

function enableBtn() {
  document.getElementById('good').disabled = false;
  document.getElementById('okay').disabled = false;
  document.getElementById('bad').disabled = false;
}

function disableBtn() {
  document.getElementById('good').disabled = true;
  document.getElementById('okay').disabled = true;
  document.getElementById('bad').disabled = true;
}

function toggleMask() {
  var checked = document.getElementById('maskcheck').checked;
  if (checked) {
    document.getElementById('mask').style.opacity = 1.0;
  } else {
    document.getElementById('mask').style.opacity = 0.0;
  }
}
