<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="images/airbnb_spain.png" alt="Aibnb locations" width="700" height="350">

  <h3 align="center">A look at Airbnb data in Spain</h3>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#premise">Premise</a></li>
        <li><a href="#execution-plan">Execution Plan</a></li>
        <li><a href="#data">Data</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setting-up-a-conda-environment">Setting up a conda environment</a></li>
        <li><a href="#file-descriptions">File descriptions</a></li>
      </ul>
    </li>
    <li><a href="#additional-notes">Additional Notes</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---
## About The Project

In this mini-project, I use the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) process to answer several business questions about Airbnb locations and reservations across Spain using their publicly-available data. Get to know the main insights by reading [my post on Medium]().

### Premise

We will take the role of a private investor that has decided to purchase a property in Spain for renting it out through Airbnb. After careful examination, we have selected 9 possible Spanish cities where it would be interesting to make such a purchase. Naturally, we want to maximize our return on investment (ROI), for which we need to understand the competition in each city as well as the main price drivers for each location.

After having a brief look at the available data, we have selected a few questions that will aid us in making our investment decisions:

  1. _What is the average price of each location type per neighbourhood? What are the most expensive neighbourhoods on average?_
  2. _What is the average host acceptance rate per location type and neighborhood? In which neighbourhoods is it the highest and in which the lowest?_
  3. _How is the competition in each neighbourhood? What number and proportion of listings belong to hosts owning different numbers of locations?_
  4. _What is the expected average profit per room type and neighborhood when looking at the reservations for the next 4 weeks? What is the neighbourhood expected to be the most profitable in that period?_
  5. _What listings' factors affect the number of reservations the most? Can we use them to forecast the number of reservations for the next 4 weeks?_

We will be comparing the answers to those questions among the different Spanish regions of **Madrid**, **Barcelona**, **Girona**, **Valencia**, **Mallorca**, **Menorca**, **Sevilla**, **Málaga** and **Euskadi**. Hopefully, this will help us in making a more informed investment decision.

<p align="right">(<a href="#top">back to top</a>)</p>

### Execution plan

In order answer our questions, we will follow the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) process. Our list of questions is already the result of the first two steps (Business Understanding and Data Understanding). We will then prepare the data as necessary to obtain the answers to our questions. This part will include performing all sorts of pre-processing steps, such as data cleaning as well as dealing with missing values. For our final question, we will also be modelling the data and try to predict the number of reservations for each location.

All processing is done with the help of Python and its widely-used libraries such as `pandas`, `numpy` and `scikit-learn`.

<p align="right">(<a href="#top">back to top</a>)</p>

### Data

This project uses [publicly-available Airbnb data](http://insideairbnb.com/get-the-data/) for 9 Spanish regions (the September 2022 version of each region). For each region, we have two different datasets:

- **Listings**: Contains all kinds of information regarding Airbnb listings, such as location, host it belongs to, type, etc. The complete data dictionary can be found in `data/airbnb/listings_schema.csv`.
- **Calendar**: Contains reservations for all listings and the price at which they were reserved.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Getting Started

To make use of this project, I recommend managing the required dependencies with Anaconda.

### Setting up a conda environment

Install miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Install mamba:

```bash
conda install -n base -c conda-forge mamba
```

Install environment using provided file:

```bash
mamba env create -f environment.yml # alternatively use environment_hist.yml if base system is not debian
mamba activate airbnb_spain
```

And finally, follow along the main notebook: `notebooks/main.ipynb`.

### File descriptions

The project files are structured as follows:

- `data/airbnb`: Where all data is located.
- `notebooks/main.ipynb`: The Jupyter notebook that runs the complete project.
- `src`: Contains the source code of helper functions used in the data wrangling and analysis.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Additional Notes

Source files formatted using the following commands:

```bash
isort .
autoflake -r --in-place --remove-unused-variable --remove-all-unused-imports --ignore-init-module-imports .
black .
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

[Carlos Uziel Pérez Malla](https://www.carlosuziel-pm.dev/)

[GitHub](https://github.com/CarlosUziel) - [Google Scholar](https://scholar.google.es/citations?user=tEz_OeIAAAAJ&hl=es&oi=ao) - [LinkedIn](https://at.linkedin.com/in/carlos-uziel-p%C3%A9rez-malla-323aa5124) - [Twitter](https://twitter.com/perez_malla)

## Acknowledgments

This project was done as part of the [Data Science Nanodegree Program at Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

<p align="right">(<a href="#top">back to top</a>)</p>
