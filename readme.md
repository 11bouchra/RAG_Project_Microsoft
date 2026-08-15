<h1>Build a RAG with Python, ChromaDB, and Ollama</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>

<h3>1. Create a virtual environment</h3>

<pre><code>python -m venv venv</code></pre>

<h3>2. Activate the virtual environment</h3>

<p><strong>Windows:</strong></p>

<pre><code>venv\Scripts\Activate</code></pre>

<p><strong>Mac/Linux:</strong></p>

<pre><code>source venv/bin/activate</code></pre>

<h3>3. Install the required libraries</h3>

<pre><code>pip install -r requirements.txt</code></pre>

<h3>4. Install Ollama locally</h3>

<p>On Windows, Ollama can be installed using:</p>

<pre><code>irm https://ollama.com/install.ps1 | iex</code></pre>

<h3>5. Pull Llama 3</h3>

<pre><code>ollama pull llama3</code></pre>

<h2>Executing the Scripts</h2>

<ul>
  <li>Open a terminal in VS Code.</li>
  <li>Execute the following commands:</li>
</ul>

<pre><code>python fill_db.py
python ask.py
python evaluate_rag.py
python run_metrics.py</code></pre>