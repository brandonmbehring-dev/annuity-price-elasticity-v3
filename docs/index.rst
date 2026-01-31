Annuity Price Elasticity v3 Documentation
=========================================

**Multi-Product Annuity Price Elasticity Analysis System**

This documentation covers the v3 architecture for estimating price sensitivity
in annuity products (RILA, FIA, MYGA).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   README

.. toctree::
   :maxdepth: 2
   :caption: Development Guides

   development/TESTING_STRATEGY

.. toctree::
   :maxdepth: 2
   :caption: Validation

   validation/VALIDATION_EVIDENCE

.. toctree::
   :maxdepth: 2
   :caption: Best Practices

   guides/COMMON_PITFALLS

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

* :doc:`development/TESTING_STRATEGY` - 6-layer validation architecture
* :doc:`validation/VALIDATION_EVIDENCE` - Proof that models are valid
* :doc:`guides/COMMON_PITFALLS` - Avoid data leakage bugs

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

**Leakage Gates**
   Automated detection of data leakage bugs including lag-0 features,
   temporal boundary violations, and scaling leakage.

Project Statistics
==================

* **Tests**: 6,126+
* **Coverage**: 70%+
* **Products**: 3 (RILA, FIA, MYGA)
* **Documentation**: 96+ files
