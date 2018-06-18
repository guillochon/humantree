function insertSrcs() {
  var ids = document.head.querySelector("[property=ids]").content.split(',');
  var i = Math.floor(Math.random() * ids.length);

  document.getElementById('satellite').src = 'parcels/' + ids[i] + '.png';
  document.getElementById('mask').src = 'parcels/' + ids[i] + '-mask.png';
}
