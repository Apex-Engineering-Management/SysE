Engineering Economics
=====================

.. currentmodule:: syse
.. autosummary::
    fv
    pmt
    nper
    ipmt
    ppmt
    pv
    rate
    irr
    npv
    mirr
    depreciate



Discounted Cash Flows
---------------------

Future (FV)
^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: fv


Payment (PMT)
^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: pmt

Number of Periodic Payments (NPER)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: nper

Interest Portion of Payment (IPMT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: ipmt

Payment Against Loan Principal (PPMT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: ppmt

Present Value (PV)
^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: pv

Rate of Interest (RATE)
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: rate

Internal Rate of Return (IRR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: irr

Net Present Value (NPV)
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: npv

Modified Internal Rate of Return (MIRR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: mirr

Evaluation of Alternatives
--------------------------

Cost Analyses
-------------

Depreciation & Taxes
--------------------


Depreciation is a non-cash expense that represents the need to replace assets eventually as they wear out.
Different types of assets have different rates at which they lose value.
This decline in value is not always linear, because some assets lose value faster earlier in their life,
and some assets hold most of their value early but then lose value faster later on. Some assets have residual value at
the end of their useful life, called salvage value.

Unless otherwise stated, the fiscal year for a company is the calendar year.

The useful life of the asset is estimated to be N years. The depreciation amount in the first year is adjusted if the
asset is purchased in the middle of the year.

The Book Value for year t is BVt , and it can not be less than the Salvage Value S.

Straight-Line Depreciation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: syse

.. toctree::

.. autofunction:: depreciate