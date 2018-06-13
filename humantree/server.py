"""Basic Flask server for HumanTree.org."""
from flask import Flask, render_template, send_from_directory
from humantree import HumanTree
app = Flask(__name__)


@app.route('/')
def index():
    """Return the index page."""
    return render_template('index.html')


@app.route('/process', methods=['GET', 'POST'])
def process():
    """Process request."""
    from flask import request

    global ht

    """Process an address."""
    if request.method == 'POST':
        address = request.form.get('address')
        print(address)
        return ht.get_image_from_address(address).split('/')[-1]


@app.route('/<path:path>')
def send_file(path):
    """Send a file to the user."""
    return send_from_directory('', path)


ht = HumanTree()
if __name__ == '__main__':
    app.run_server(debug=True)
