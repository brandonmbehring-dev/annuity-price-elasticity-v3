Annuity Price Elasticity v3 Documentation
=========================================

**Multi-Product Annuity Price Elasticity Analysis System**

This documentation covers the v3 architecture for estimating price sensitivity
in annuity products (RILA, FIA, MYGA).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   README
   onboarding/GETTING_STARTED
   onboarding/FIRST_MODEL_GUIDE

.. toctree::
   :maxdepth: 2
   :caption: Analysis

   analysis/CAUSAL_FRAMEWORK
   analysis/FEATURE_INTERPRETATION
   analysis/MODEL_INTERPRETATION

.. toctree::
   :maxdepth: 2
   :caption: Development Guides

   development/TESTING_GUIDE
   development/CODING_STANDARDS
   guides/TROUBLESHOOTING

.. toctree::
   :maxdepth: 2
   :caption: Best Practices

   practices/LEAKAGE_CHECKLIST
   practices/ANTI_PATTERNS
   integration/LESSONS_LEARNED

.. toctree::
   :maxdepth: 2
   :caption: Bug Postmortems

   knowledge/episodes/episode_01_lag0_competitor_rates
   knowledge/episodes/episode_02_aggregation_lookahead
   knowledge/episodes/episode_03_feature_selection_bias
   knowledge/episodes/episode_04_product_mix_confounding
   knowledge/episodes/episode_05_market_weight_leakage
   knowledge/episodes/episode_06_temporal_cv_violation
   knowledge/episodes/episode_07_scaling_leakage
   knowledge/episodes/episode_08_holiday_lookahead
   knowledge/episodes/episode_09_macro_lookahead
   knowledge/episodes/episode_10_own_rate_endogeneity

.. toctree::
   :maxdepth: 2
   :caption: Domain Knowledge

   domain-knowledge/RILA_ECONOMICS
   domain-knowledge/GLOSSARY

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Links
===========

* :doc:`analysis/FEATURE_INTERPRETATION` - Feature coefficient meanings
* :doc:`practices/LEAKAGE_CHECKLIST` - Pre-deployment gate
* :doc:`guides/TROUBLESHOOTING` - Pain-point organized debugging

Key Features
============

**Multi-Product Architecture**
   Support for RILA (6Y20B, 6Y10B, 10Y20B), FIA, and MYGA products
   with product-specific methodologies.

**Dependency Injection Pattern**
   Clean separation between data sources (AWS, Local, Fixture) and
   business logic via adapters.

**6-Layer Validation**
   Comprehensive testing from unit tests to end-to-end validation
   with automated leakage detection.

**Known-Answer Tests**
   Validation against LIMRA 2023 literature bounds and golden reference
   values for regression detection.

**10 Bug Postmortems**
   Complete audit trail of data leakage categories with symptoms,
   root causes, and prevention gates.

Project Statistics
==================

* **Tests**: 6,200+
* **Coverage**: 70%+
* **Products**: 3 (RILA, FIA, MYGA)
* **Documentation**: 111+ files
* **Bug Episodes**: 10
