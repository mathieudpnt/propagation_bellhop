<h2>Overview</h2>
<p>This repository contains the Python-based simulator developed by <a href="https://maree.fr/">maree</a> as part of a study on bioacoustic detection of cetaceans in the Iroise Marine Park. The goal of this simulator is to evaluate the detection distances of cetacean signals and help users understand how these signals propagate based on environmental factors.</p>

<h2>Context</h2>
<p>The study was initiated by bioacousticians at ENSTA, who deployed seven acoustic recorders in the Iroise Marine Park for one year (<a href="https://osmose.ifremer.fr/projects/1">CETIROISE project</a>). The collected data allowed for the detection and identification of dolphins and porpoises, showing seasonal and location-based variations in detection rates. A key challenge is understanding the representativity of these detections relative to actual cetacean presence, which is influenced by the propagation characteristics of their acoustic signals.</p>

<h2>Features</h2>
<ul>
    <li><strong>Simulates Cetacean Signal Propagation:</strong> Models how high-frequency signals, shallow immersions, and directional emissions affect detection distances.</li>
    <li><strong>Environmental Parameter Integration:</strong> Considers factors such as water depth, temperature, and salinity.</li>
    <li><strong>Bellhop Propagation Model:</strong> Signal propagation is calculated using the Bellhop code, based on ray theory. The simulation assumes a variable bathymetry, with a homogeneous and fluid seabed. The bathymetric speed is treated as constant relative to the source position.</li>
    <li><strong>Visualization Tools:</strong> Provides plots and analysis tools for better interpretation of results.</li>
</ul>

<h3>Clone the Repository</h3>
<pre><code>git clone https://github.com/mathieudpnt/propagation_bellhop.git
</code></pre>

<h2>Citation</h2>
<p>The Bellhop program used for signal propagation is part of the Acoustics Toolbox and can be downloaded from <a href="https://oalib-acoustics.org/website_resources/AcousticsToolbox/versions/at/" target="_blank">Acoustics Toolbox</a>. Please cite the following reference when using this software:</p>
<blockquote>
  <p><strong>Bellhop</strong> (2020). <em>Bellhop Acoustics Toolbox</em>. Retrieved from <a href="https://oalib-acoustics.org/website_resources/AcousticsToolbox/" target="_blank">https://oalib-acoustics.org/website_resources/AcousticsToolbox/</a>.</p>
</blockquote>

<h2>Contributing</h2>
<p>Contributions to this repository are welcome! If you would like to contribute, please follow the steps below:</p>
<ol>
  <li>Fork the repository.</li>
  <li>Clone your fork to your local machine.</li>
  <li>Create a new branch for your changes.</li>
  <li>Make your changes, ensuring that your code is properly tested.</li>
  <li>Commit your changes and push them to your fork.</li>
  <li>Submit a pull request from your fork's branch to the main repository.</li>
</ol>
<p>When submitting a pull request, please ensure your code adheres to the existing code style and includes relevant tests. Provide a clear description of what your changes do and why they are necessary. Your contribution is appreciated!</p>
<br>
<div style="text-align: center;">
  <img src="https://img.shields.io/pypi/status/ansicolortags.svg" alt="PyPI status" />
  <img src="https://img.shields.io/github/license/mashape/apistatus.svg" alt="license" />
  <img src="https://img.shields.io/badge/open%20source-♡-lightgrey" alt="Open Source Love" />
</div>

