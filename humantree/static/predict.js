/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http:// www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==============================================================================
*/

// This tiny example illustrates how little code is necessary build /
// train / predict from a model in TensorFlow.js.  Edit this code
// and refresh the index.html to quickly explore the API.

// Tiny TFJS train / predict example.
async function loadModel() {
  // Load model.
  console.log('Loading model...');
  this.model = await tf.loadModel('assets/model.json');
  console.log('Model loaded.');
}

async function predict(imageData) {

  await tf.tidy(() => {

    // Convert the canvas pixels to a Tensor of the matching shape
    let img = tf.fromPixels(imageData, 1);
    img = img.reshape([1, 512, 512, 1]);
    img = tf.cast(img, 'float32');

    // Make and format the predications
    const output = this.model.predict(img);

    // Save predictions on the component
    this.predictions = Array.from(output.dataSync());

    document.getElementById('predict').innerText = this.predictions;
  });

}

$(function () {
    $('#process').submit(function (e) {
        console.log('hello');
        e.preventDefault();  // prevent the form from 'submitting'

        var url = e.target.action;  // get the target
        var formData = $(this).serialize(); // get form data
        $.post(url, formData, function (response) { // send; response.data will be what is returned
            console.log(url);
            console.log(response);
            $("#image").attr("src", "queries/" + response);
        });
    })
})

// function processImage() {
//     var xhr = new XMLHttpRequest();
//     xhr.onreadystatechange = function(){
//       var fname = xhr.responseText;
//       document.getElementById("image").src = fname;
//       return false;
//     } // success case
//     xhr.onerror = function(){ alert (xhr.responseText); } // failure case
//     xhr.open (oFormElement.method, oFormElement.action, true);
//     xhr.setRequestHeader("Content-type","application/x-www-form-urlencoded");
//     xhr.send (new FormData (oFormElement));
//     return false;
// }

loadModel();
