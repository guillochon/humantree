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
  var tids_str = document.head.querySelector("[property=tids]").content;
  var qids_str = document.head.querySelector("[property=qids]").content;
  var pids_str = document.head.querySelector("[property=pids]").content;
  var tids = tids_str.split(',');
  var qids = qids_str.split(',');
  var pids = pids_str.split(',');
  var ra = Math.random();
  console.log(ra);
  if ((ra < (1.0 / 3.0)) || ((qids_str == '') && (pids_str == ''))) {
    var idir = 'parcels';
    var odir = 'parcels';
    var ids = tids;
  } else if (((ra < (2.0 / 3.0)) || (qids_str == '')) && pids_str != '') {
    var idir = 'parcels';
    var odir = 'preds';
    var ids = pids;
  } else {
    var idir = 'queries';
    var odir = 'queries';
    var ids = qids;
  }
  var iids = document.head.querySelector("[property=iids]").content;

  var i = Math.floor(Math.random() * ids.length);
  var idi = ids[i]
  var idis = idi.substring(1);
  document.getElementById('satellite').src = idir + '/' + idis + '.png';
  document.getElementById('mask').src = odir + '/' + idis + '-outline.png';
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
