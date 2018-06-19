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
  var ids = document.head.querySelector("[property=ids]").content.split(',');
  var i = Math.floor(Math.random() * ids.length);

  document.getElementById('satellite').src = 'parcels/' + ids[i] + '.png';
  document.getElementById('mask').src = 'parcels/' + ids[i] + '-outline.png';
  document.getElementById('mask-num').value = ids[i];
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
