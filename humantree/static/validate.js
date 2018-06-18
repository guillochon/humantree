function insertSrcs() {
  var ids = document.head.querySelector("[property=ids]").content.split(',');
  var i = Math.floor(Math.random() * ids.length);

  document.getElementById('satellite').src = 'parcels/' + ids[i] + '.png';
  document.getElementById('mask').src = 'parcels/' + ids[i] + '-mask.png';
  document.getElementById('mask-num').value = ids[i];
}

function enableBtn() {
  document.getElementById("good").disabled = false;
  document.getElementById("bad").disabled = false;
}

function disableBtn() {
  document.getElementById("good").disabled = true;
  document.getElementById("bad").disabled = true;
}
