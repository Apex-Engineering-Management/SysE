"""Systems & Industrial Engineering functions

functions behave like ufuncs with
broadcasting and being able to be called with scalars
or arrays (or other sequences).

Functions support the :class:`decimal.Decimal` type unless
otherwise stated.
"""
from __future__ import division, absolute_import, print_function

from decimal import Decimal

import math

import numpy as np


__all__ = ['fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate',
           'irr', 'npv', 'mirr', 'depreciate', 'digits', 'decline', 'double', 'units',
           'line', 'sma', 'wma', 'linear_pro', 'pert', 'eoq', 'emq']

_when_to_num = {'end': 0, 'begin': 1,
                'e': 0, 'b': 1,
                0: 0, 1: 1,
                'beginning': 1,
                'start': 1,
                'finish': 0}


def _convert_when(when):
    # Test to see if when has already been converted to ndarray
    # This will happen if one function calls another, for example ppmt
    if isinstance(when, np.ndarray):
        return when
    try:
        return _when_to_num[when]
    except (KeyError, TypeError):
        return [_when_to_num[x] for x in when]


# Engineering Economics


# Future Value (FV)

def fv(rate, nper, pmt, pv, when='end'):
    """
    Compute the future value.

    Given:
     * a present value, `pv`
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * a (fixed) payment, `pmt`, paid either
     * at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period
    Return:
       the value at the end of the `nper` periods

    Parameters:
        rate (scalar or array_like of shape(M, )): Rate of interest as decimal (not per cent) per period
        nper (scalar or array_like of shape(M, )): Number of compounding periods
        pmt (scalar or array_like of shape(M, )): Payment
        pv (scalar or array_like of shape(M, )): Present value
        when ({{'begin', 1}, {'end', 0}}, {string, int}, optional): When payments are due ('begin' (1) or 'end' (0)).
            Defaults to {'end', 0}.

    Returns:
        ndarray: Future values. If all input is scalar, returns a scalar float. If any input is array_like, returns
        future values for each input element. If multiple inputs are array_like, they all must have the same shape.


    .. Note::
           The future value is computed by solving the equation::
                      fv +
                      pv*(1+rate)**nper +
                      pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0
           or, when ``rate == 0``::
                      fv + pv + pmt * nper == 0



    Examples:
    ---------
    You invest $20,000 in a retirement account and expect to earn a 10% annual return. How much do you expect to be in
    the account after 20 years?
    ::
        import numpy as np
        import syse as syse
        account_value = syse.fv(0.1,20,0,20000)
        print(f"Account Value = ${abs(account_value):,.2f}")
        Output = Account Value = $134,550.00
        
        
    You are planning for your retirement and will be investing $250/month into an IRA. You expect a monthly return of
    1% and are 45 years from your expected retirement date. Given this information, how much do you expect to be in your
    retirement account when you retire?
    ::
        import numpy as np
        import syse as syse
        ira_value = syse.fv(0.01,12*45,250,0)
        print(f"IRA Value = ${abs(Q5):,.2f}")
        Output = IRA Value = $5,363,673.26
    """
    when = _convert_when(when)
    rate, nper, pmt, pv, when = np.broadcast_arrays(rate, nper, pmt, pv, when)

    fv_array = np.empty_like(rate)
    zero = rate == 0
    nonzero = ~zero

    fv_array[zero] = -(pv[zero] + pmt[zero] * nper[zero])

    rate_nonzero = rate[nonzero]
    temp = (1 + rate_nonzero)**nper[nonzero]
    fv_array[nonzero] = (
        - pv[nonzero] * temp
        - pmt[nonzero] * (1 + rate_nonzero * when[nonzero]) / rate_nonzero
        * (temp - 1)
    )

    if np.ndim(fv_array) == 0:
        # Follow the ufunc convention of returning scalars for scalar
        # and 0d array inputs.
        return fv_array.item(0)
    return fv_array


# Payment Against Loan Principal Plus Interest

def pmt(rate, nper, pv, fv=0, when='end'):
    """
    Compute the payment against loan principal plus interest.

    Given:
     * a present value, `pv` (e.g., an amount borrowed)
     * a future value, `fv` (e.g., 0)
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * and (optional) specification of whether payment is made
       at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period
    Return:
       the (fixed) periodic payment.

    Parameters:
        rate (array_like): Rate of interest (per period)
        nper (array_like): Number of compounding periods
        pv (array_like): Present value
        fv (array_like, optional): Future value (default = 0)
        when ({{'begin', 1}, {'end', 0}}, {string, int}): When payments are due ('begin' (1) or 'end' (0))

    Returns:
    --------
        ndarray: Payment against loan plus interest. If all input is scalar, returns a scalar float. If any input is
        array_like, returns payment for each input element. If multiple inputs are array_like, they all must have
        the same shape.

    .. Note::
        The payment is computed by solving the equation::
            fv +
            pv*(1 + rate)**nper +
            pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0
        or, when ``rate == 0``::
            fv + pv + pmt * nper == 0
        for ``pmt``.
        Note that computing a monthly mortgage payment is only
        one use for this function.  For example, pmt returns the
        periodic deposit one must make to achieve a specified
        future balance given an initial deposit, a fixed,
        periodically compounded interest rate, and the total
        number of periods.

    Examples:
    ---------
    What is the monthly payment needed to pay off a $200,000 loan in 15
    years at an annual interest rate of 7.5%?
    ::
        import syse as syse
        syse.pmt(0.075/12, 12*15, 200000)
        Output = -1854.0247200054619
    In order to pay-off (i.e., have a future-value of 0) the $200,000 obtained
    today, a monthly payment of $1,854.02 would be required.  Note that this
    example illustrates usage of `fv` having a default value of 0.
    """
    when = _convert_when(when)
    (rate, nper, pv, fv, when) = map(np.array, [rate, nper, pv, fv, when])
    temp = (1 + rate)**nper
    mask = (rate == 0)
    masked_rate = np.where(mask, 1, rate)
    fact = np.where(mask != 0, nper,
                    (1 + masked_rate*when)*(temp - 1)/masked_rate)
    return -(fv + pv*temp) / fact


# Number of Periodic Payments

def nper(rate, pmt, pv, fv=0, when='end'):
    """
    Compute the number of periodic payments.

    :class:`decimal.Decimal` type is not supported.

    Parameters:
        rate (array_like): Rate of interest (per period)
        pmt (array_like): Payment
        pv (array_like): Present value
        fv (array_like, optional): Future value
        when ({{'begin', 1}, {'end', 0}}, {string, int}, optional): When payments are due ('begin' (1) or 'end' (0))

    .. Note::
        The number of periods ``nper`` is computed by solving the equation::
            fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate*((1+rate)**nper-1) = 0
        but if ``rate = 0`` then::
            fv + pv + pmt*nper = 0

    Examples:
    ---------
    If you only had $150/month to pay towards the loan, how long would it take
    to pay-off a loan of $8,000 at 7% annual interest?
    ::
        import numpy as np
        import syse as syse
        print(np.round(syse.nper(0.07/12, -150, 8000), 5))
        Output = 64.07335
    So, over 64 months would be required to pay off the loan.
    The same analysis could be done with several different interest rates
    and/or payments and/or total amounts to produce an entire table::
        syse.nper(*(np.ogrid[0.07/12: 0.08/12: 0.01/12,
                            -150: -99: 50,
                            8000: 9001: 1000]))
        Output = array([[[ 64.07334877,  74.06368256],
                 [108.07548412, 127.99022654]],
                 [[ 66.12443902,  76.87897353],
                 [114.70165583, 137.90124779]]])
    """
    when = _convert_when(when)
    rate, pmt, pv, fv, when = np.broadcast_arrays(rate, pmt, pv, fv, when)
    nper_array = np.empty_like(rate, dtype=np.float64)

    zero = rate == 0
    nonzero = ~zero

    with np.errstate(divide='ignore'):
        # Infinite numbers of payments are okay, so ignore the
        # potential divide by zero.
        nper_array[zero] = -(fv[zero] + pv[zero]) / pmt[zero]

    nonzero_rate = rate[nonzero]
    z = pmt[nonzero] * (1 + nonzero_rate * when[nonzero]) / nonzero_rate
    nper_array[nonzero] = (
        np.log((-fv[nonzero] + z) / (pv[nonzero] + z))
        / np.log(1 + nonzero_rate)
    )

    return nper_array


def _value_like(arr, value):
    entry = arr.item(0)
    if isinstance(entry, Decimal):
        return Decimal(value)
    else:
        return np.array(value, dtype=arr.dtype).item(0)


# Interest Portion of Payment

def ipmt(rate, per, nper, pv, fv=0, when='end'):
    """
    Compute the interest portion of a payment.

    Parameters:
        rate (scalar or array_like of shape(M, )): Rate of interest as decimal (not per cent) per period
        per (scalar or array_like of shape(M, )): Interest paid against the loan changes during the life or the loan.
            The `per` is the payment period to calculate the interest amount.
        nper (scalar or array_like of shape(M, )): Number of compounding periods
        pv (scalar or array_like of shape(M, )): Present value
        fv (scalar or array_like of shape(M, ), optional): Future value
        when ({{'begin', 1}, {'end', 0}}, {string, int}, optional): When payments are due ('begin' (1) or 'end' (0)).
            Defaults to {'end', 0}.

    Returns:
        ndarray: Interest portion of payment. If all input is scalar, returns a scalar float. If any input is
        array_like, returns interest payment for each input element. If multiple inputs are array_like,
        they all must have the same shape.


    See Also
    --------
    ppmt, pmt, pv




    .. Note::
        The total payment is made up of payment against principal plus interest.
        ``pmt = ppmt + ipmt``



    Examples:
    ---------
    What is the amortization schedule for a 1 year loan of $2500 at
    8.24% interest per year compounded monthly?
    ::
        import numpy as np
        import syse as syse
        principal = 2500.00
    The 'per' variable represents the periods of the loan. Remember that financial equations start the period count at 1!
    ::
        per = np.arange(1*12) + 1
        ipmt = syse.ipmt(0.0824/12, per, 1*12, principal)
        ppmt = syse.ppmt(0.0824/12, per, 1*12, principal)
    Each element of the sum of the 'ipmt' and 'ppmt' arrays should equal 'pmt'.
    ::
        pmt = syse.pmt(0.0824/12, 1*12, principal)
        np.allclose(ipmt + ppmt, pmt)
        True
    ::
        fmt = '{0:2d} {1:8.2f} {2:8.2f} {3:8.2f}'
        for payment in per:
        index = payment - 1
        principal = principal + ppmt[index]
        print(fmt.format(payment, ppmt[index], ipmt[index], principal))
        1  -200.58   -17.17  2299.42
        2  -201.96   -15.79  2097.46
        3  -203.35   -14.40  1894.11
        4  -204.74   -13.01  1689.37
        5  -206.15   -11.60  1483.22
        6  -207.56   -10.18  1275.66
        7  -208.99    -8.76  1066.67
        8  -210.42    -7.32   856.25
        9  -211.87    -5.88   644.38
        10  -213.32    -4.42   431.05
        11  -214.79    -2.96   216.26
        12  -216.26    -1.49    -0.00
        interestpd = np.sum(ipmt)
        np.round(interestpd, 2)
        -112.98
    """
    when = _convert_when(when)
    rate, per, nper, pv, fv, when = np.broadcast_arrays(rate, per, nper,
                                                        pv, fv, when)

    total_pmt = pmt(rate, nper, pv, fv, when)
    ipmt_array = np.array(_rbl(rate, per, total_pmt, pv, when) * rate)

    # Payments start at the first period, so payments before that
    # don't make any sense.
    ipmt_array[per < 1] = _value_like(ipmt_array, np.nan)
    # If payments occur at the beginning of a period and this is the
    # first period, then no interest has accrued.
    per1_and_begin = (when == 1) & (per == 1)
    ipmt_array[per1_and_begin] = _value_like(ipmt_array, 0)
    # If paying at the beginning we need to discount by one period.
    per_gt_1_and_begin = (when == 1) & (per > 1)
    ipmt_array[per_gt_1_and_begin] = (
        ipmt_array[per_gt_1_and_begin] / (1 + rate[per_gt_1_and_begin])
    )

    if np.ndim(ipmt_array) == 0:
        # Follow the ufunc convention of returning scalars for scalar
        # and 0d array inputs.
        return ipmt_array.item(0)
    return ipmt_array


def _rbl(rate, per, pmt, pv, when):
    """
    This function is here to simply have a different name for the 'fv'
    function to not interfere with the 'fv' keyword argument within the 'ipmt'
    function.  It is the 'remaining balance on loan' which might be useful as
    it's own function, but is easily calculated with the 'fv' function.
    """
    return fv(rate, (per - 1), pmt, pv, when)


# Payment Against Loan Principal

def ppmt(rate, per, nper, pv, fv=0, when='end'):
    """
    Compute the payment against loan principal.

    Parameters:
        rate (array_like): Rate of interest (per period)
        per (array_like, int): Amount paid against the loan changes. The `per` is the period of interest.
        nper (array_like): Number of compounding periods
        pv (array_like): Present value
        fv (array_like, optional): Future value
        when ({{'begin', 1}, {'end', 0}}, {string, int}): When payments are due ('begin' (1) or 'end' (0))
    See Also
    --------
    pmt, pv, ipmt
    """
    total = pmt(rate, nper, pv, fv, when)
    return total - ipmt(rate, per, nper, pv, fv, when)


# Present Value

def pv(rate, nper, pmt, fv=0, when='end'):
    """
    Compute the present value.

    Given:
     * a future value, `fv`
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * a (fixed) payment, `pmt`, paid either
     * at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period

    Return:
       the value now

    Parameters:
        rate (array_like): Rate of interest (per period)
        nper (array_like): Number of compounding periods
        pmt (array_like): Payment
        fv (array_like, optional): Future value
        when ({{'begin', 1}, {'end', 0}}, {string, int}, optional): When payments are due ('begin' (1) or 'end' (0))

    Returns:
        ndarray, float: Present value of a series of payments or investments.

    .. Note::
        The present value is computed by solving the equation::
            fv + pv*(1 + rate)**nper + pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) = 0
        or, when ``rate = 0``::
            fv + pv + pmt * nper = 0
        for `pv`, which is then returned.

    Examples:
    ---------
    A German bond that pays annual coupons of 4.5%, has a par value of €1,000, and a YTM of 3.9%.
    Assuming there are 19 years until maturity, what is the current price of this bond?
    ::
        import numpy as np
        import syse as syse
        face_value=1000
        coupon_value=4.5
        num_coupons=19
        ytm=3.9
        current_price = syse.pv(0.039, 19, 45, 1000)
        print(f"Current price = €{abs(current_price):,.2f}")
        Output = Current price = €1,079.48

    Gruber Corp. pays a constant $9 dividend on its stock. The company will maintain this dividend for the next 12
    years and will then cease paying dividends forever (you should assume the company disappears with no extra payouts).
    If the required return on this stock is 10%, what is the current share price?
    ::
        import numpy as np
        import syse as syse
        rate = 0.10
        nper = 12
        pmt = 9
        share_price = syse.pv(0.10, 12, 9)
        print(f"Share Price = ${abs(share_price):.2f}")
        Output = Share Price = $61.32
    """
    when = _convert_when(when)
    (rate, nper, pmt, fv, when) = map(np.asarray, [rate, nper, pmt, fv, when])
    temp = (1+rate)**nper
    fact = np.where(rate == 0, nper, (1+rate*when)*(temp-1)/rate)
    return -(fv + pmt*fact)/temp

# Computed with Sage
#  (y + (r + 1)^n*x + p*((r + 1)^n - 1)*(r*w + 1)/r)/(n*(r + 1)^(n - 1)*x -
#  p*((r + 1)^n - 1)*(r*w + 1)/r^2 + n*p*(r + 1)^(n - 1)*(r*w + 1)/r +
#  p*((r + 1)^n - 1)*w/r)


def _g_div_gp(r, n, p, x, y, w):
    # Evaluate g(r_n)/g'(r_n), where g =
    # fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1)
    t1 = (r+1)**n
    t2 = (r+1)**(n-1)
    g = y + t1*x + p*(t1 - 1) * (r*w + 1) / r
    gp = (n*t2*x
          - p*(t1 - 1) * (r*w + 1) / (r**2)
          + n*p*t2 * (r*w + 1) / r
          + p*(t1 - 1) * w/r)
    return g / gp

# Use Newton's iteration until the change is less than 1e-6
#  for all values or a maximum of 100 iterations is reached.
#  Newton's rule is
#  r_{n+1} = r_{n} - g(r_n)/g'(r_n)
#     where
#  g(r) is the formula
#  g'(r) is the derivative with respect to r.


# Rate of Interest

def rate(nper, pmt, pv, fv, when='end', guess=None, tol=None, maxiter=100):
    """
    Compute the rate of interest per period.

    Parameters:
        nper (array_like): Number of compounding periods
        pmt (array_like): Payment
        pv (array_like): Present value
        fv (array_like): Future value
        when ({{'begin', 1}, {'end', 0}}, {string, int}, optional): When payments are due ('begin' (1) or 'end' (0))
        guess (Number, optional): Starting guess for solving the rate of interest, default 0.1
        tol (Number, optional): Required tolerance for the solution, default 1e-6
        maxiter (int, optional): Maximum iterations in finding the solution

    .. Note::
        The rate of interest is computed by iteratively solving the
        (non-linear) equation::
            fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) = 0
        for ``rate``.
    """
    when = _convert_when(when)
    default_type = Decimal if isinstance(pmt, Decimal) else float

    # Handle casting defaults to Decimal if/when pmt is a Decimal and
    # guess and/or tol are not given default values
    if guess is None:
        guess = default_type('0.1')

    if tol is None:
        tol = default_type('1e-6')

    (nper, pmt, pv, fv, when) = map(np.asarray, [nper, pmt, pv, fv, when])

    rn = guess
    iterator = 0
    close = False
    while (iterator < maxiter) and not np.all(close):
        rnp1 = rn - _g_div_gp(rn, nper, pmt, pv, fv, when)
        diff = abs(rnp1-rn)
        close = diff < tol
        iterator += 1
        rn = rnp1

    if not np.all(close):
        if np.isscalar(rn):
            return default_type(np.nan)
        else:
            # Return nan's in array of the same shape as rn
            # where the solution is not close to tol.
            rn[~close] = np.nan
    return rn


# Internal Rate of Return (IRR)

def irr(values, guess=0.1, tol=1e-12, maxiter=100):
    """
    Return the Internal Rate of Return (IRR).
    This is the "average" periodically compounded rate of return
    that gives a net present value of 0.0; for a more complete explanation,
    see Notes below.

    :class:`decimal.Decimal` type is not supported.

    Parameters:
        values (array_like, shape(N,)): Input cash flows per time period. By convention, net "deposits" are negative
            and net "withdrawals" are positive. Thus, for example, at least the first element of `values`, which
            represents the initial investment, will typically be negative.
        guess (float, optional): Initial guess of the IRR for the iterative solver. If no guess is given an initial
            guess of 0.1 (i.e. 10%) is assumed instead.
        tol (float, optional): Required tolerance to accept solution. Default is 1e-12.
        maxiter (int, optional): Maximum iterations to perform in finding a solution. Default is 100.

    Returns:
        float: Internal Rate of Return for periodic input values.

    .. Note::
        The IRR is perhaps best understood through an example (illustrated
        using np.irr in the Examples section below). Suppose one invests 100
        units and then makes the following withdrawals at regular (fixed)
        intervals: 39, 59, 55, 20.  Assuming the ending value is 0, one's 100
        unit investment yields 173 units; however, due to the combination of
        compounding and the periodic withdrawals, the "average" rate of return
        is neither simply 0.73/4 nor `(1.73)^(0.25-1)`.  Rather, it is the solution
        (for :math:`r`) of the equation:
    .. Math::
        -100 + \\frac{39}{1+r} + \\frac{59}{(1+r)^2} + \\frac{55}{(1+r)^3} + \\frac{20}{(1+r)^4} = 0
    .. code::
       -100 + (39/1+r) = 0
    In general, for `values` :math:`= [v_0, v_1, ... v_M]`,
    irr is the solution of the equation: [G]_
    .. math::
        \\sum_{t=0}^M{\\frac{v_t}{(1+irr)^{t}}} = 0

    Examples:
    ---------
    ::
        import syse as syse
        round(syse.irr([-100, 39, 59, 55, 20]), 5)
        0.28095
        round(syse.irr([-100, 0, 0, 74]), 5)
        -0.0955
        round(syse.irr([-100, 100, 0, -7]), 5)
        -0.0833
        round(syse.irr([-100, 100, 0, 7]), 5)
        0.06206
        round(syse.irr([-5, 10.5, 1, -8, 1]), 5)
        0.0886
    """
    values = np.atleast_1d(values)
    if values.ndim != 1:
        raise ValueError("Cashflows must be a rank-1 array")

    # If all values are of the same sign no solution exists
    # we don't perform any further calculations and exit early
    same_sign = np.all(values > 0) if values[0] > 0 else np.all(values < 0)
    if same_sign:
        return np.nan

    # We aim to solve eirr such that NPV is exactly zero. This can be framed as
    # simply finding the closest root of a polynomial to a given initial guess
    # as follows:
    #           V0           V1           V2           V3
    # NPV = ---------- + ---------- + ---------- + ---------- + ...
    #       (1+eirr)^0   (1+eirr)^1   (1+eirr)^2   (1+eirr)^3
    #
    # by letting x = 1 / (1+eirr), we substitute to get
    #
    # NPV = V0 * x^0   + V1 * x^1   +  V2 * x^2  +  V3 * x^3  + ...
    #
    # which we solve using Newton-Raphson and then reverse out the solution
    # as eirr = 1/x - 1 (if we are close enough to a solution)
    npv_ = np.polynomial.Polynomial(values)
    d_npv = npv_.deriv()
    x = 1 / (1 + guess)

    for _ in range(maxiter):
        x_new = x - (npv_(x) / d_npv(x))
        if abs(x_new - x) < tol:
            return 1 / x_new - 1
        x = x_new

    return np.nan


# Net Present Value (NPV)

def npv(rate, values):
    """
    Returns the NPV (Net Present Value) of a cash flow series.

    Parameters:
        rate (scalar): The discount rate.
        values (array_like, shape(M, )): The values of the time series of cash flows. The (fixed) time interval
            between cash flow "events" must be the same as that for which `rate` is given (i.e., if `rate` is per year,
            then precise a year is understood to elapse between each cash flow event). By convention, investments or
            "deposits" are negative, income or "withdrawals" are positive; `values` must begin with the initial
            investment, thus `values[0]` will typically be negative.

    Returns:
        float: The NPV of the input cash flow series `values` at the discount `rate`.

    Warnings
    --------
    ``npv`` considers a series of cashflows starting in the present (t = 0).
    NPV can also be defined with a series of future cashflows, paid at the
    end, rather than the start, of each period. If future cashflows are used,
    the first cashflow `values[0]` must be zeroed and added to the net
    present value of the future cashflows. This is demonstrated in the
    examples.

    Notes
    -----
    Returns the result of: [G]_
    .. math:: \\sum_{t=0}^{M-1}{\\frac{values_t}{(1+rate)^{t}}}

    Examples:
    ---------
    Consider a potential project with an initial investment of $40 000 and
    projected cashflows of $5 000, $8 000, $12 000 and $30 000 at the end of
    each period discounted at a rate of 8% per period. To find the project's
    net present value::
        import numpy as np
        import syse as syse
        rate, cashflows = 0.08, [-40_000, 5_000, 8_000, 12_000, 30_000]
        syse.npv(rate, cashflows).round(5)
        Output = 3065.22267
    It may be preferable to split the projected cashflow into an initial
    investment and expected future cashflows. In this case, the value of
    the initial cashflow is zero and the initial investment is later added
    to the future cashflows net present value::
        initial_cashflow = cashflows[0]
        cashflows[0] = 0
        np.round(syse.npv(rate, cashflows) + initial_cashflow, 5)
        Output = 3065.22267
    """
    values = np.atleast_2d(values)
    timestep_array = np.arange(0, values.shape[1])
    npv = (values / (1 + rate) ** timestep_array).sum(axis=1)
    try:
        # If size of array is one, return scalar
        return npv.item()
    except ValueError:
        # Otherwise, return entire array
        return npv


# Modified Internal Rate of Return

def mirr(values, finance_rate, reinvest_rate):
    """
    Modified internal rate of return.

    Parameters:
        values (array_like): Cash flows (must contain at least one positive and one negative value) or nan is returned.
            The first value is considered a sunk cost at time zero.
        finance_rate (scalar): Interest rate paid on the cash flows
        reinvest_rate (scalar): Interest rate received on the cash flows upon reinvestment

    Returns:
        float: Modified internal rate of return
    """
    values = np.asarray(values)
    n = values.size

    # Without this explicit cast the 1/(n - 1) computation below
    # becomes a float, which causes TypeError when using Decimal
    # values.
    if isinstance(finance_rate, Decimal):
        n = Decimal(n)

    pos = values > 0
    neg = values < 0
    if not (pos.any() and neg.any()):
        return np.nan
    numer = np.abs(npv(reinvest_rate, values*pos))
    denom = np.abs(npv(finance_rate, values*neg))
    return (numer/denom)**(1/(n - 1))*(1 + reinvest_rate) - 1

# Depreciation & Taxes


# Straight-Line Depreciation

def depreciate(cost, salvage, life):
    """
    Calculate the straight-line depreciation of an asset over time.

    Parameters:
        cost: initial cost of the asset
        salvage: salvage value of the asset at the end of its useful life
        life: life of the asset in years

    Returns:
        float: depreciation amount

    .. Note::
        The depreciation amount for each full year is the same amount:
           the original value of the asset B minus the salvage value S all divided by the number of years N

    Examples:
    ---------
    A company purchased a machine for $100,000 with an estimated salvage value of $10,000 after 5 years::
        cost = 100000
        salvage = 10000
        life = 5
        syse.depreciate(cost, salvage, life)
        print("The straight-line depreciation for the machine is ${} per year.".format(depreciate))
        Output = The straight-line depreciation for the machine is $18,000 per year.

    .. jupyter-execute::
        :hide-code:
        cost = 100000
        salvage = 10000
        life = 5
        depreciation = (cost - salvage) / life
        print("The straight-line depreciation for the machine is ${} per year.".format(depreciation))
    """
    depreciation = (cost - salvage) / life
    return depreciation


# Sum-of-Years-Digits Method:

def digits(cost: np.ndarray, salvage_value: np.ndarray, useful_life: int) -> np.ndarray:
    """
    Compute the depreciation for an asset using the sum-of-years-digits method.

    Parameters:
        cost (np.ndarray): The cost of the asset.
        salvage_value (np.ndarray): The salvage value of the asset.
        useful_life (int): The useful life of the asset.

    Returns:
        np.ndarray: The depreciation for the asset.

    .. Note::
        The function takes as input the cost of the asset, the salvage_value of the asset at the end of its useful
        life, and the useful_life of the asset in years. It computes the depreciation for the asset using the
        sum-of-years-digits method, which assumes that the asset depreciates more rapidly in the earlier years of
        its life.

    Examples:
    ---------
    Suppose a company purchases a machine for $60,000, with an expected salvage value of $6,000 after 6 years.
    The company expects to use the machine for 6 years. We can compute the accumulated depreciation for the machine
    using the digits function::
        cost= 60000
        salvage_value= 6000
        useful_life= 6
        syse.digits(cost, salvage_value, useful_life)
        Output = 54000
    This means that the accumulated depreciation is $54,000 and the book value of the machine will be $6,000,
    which is the expected salvage value.
    """
    years = np.array([i + 1 for i in range(useful_life)], dtype=np.float64)
    total_years = np.sum(years)
    accumulated_depreciation = np.zeros_like(cost, dtype=np.float64)

    for year in years:
        depreciation = (cost - salvage_value) * (useful_life - year + 1) / total_years
        accumulated_depreciation += depreciation.astype(np.float64)

    return accumulated_depreciation


# Declining Balance Method:

def decline(cost: float, salvage_value: float, useful_life: int, rate: float) -> float:
    """
    Compute the depreciation for an asset using the declining balance method.

    Parameters:
        cost (float): The cost of the asset.
        salvage_value (float): The salvage value of the asset.
        useful_life (int): The useful life of the asset.
        rate (float): The depreciation rate, expressed as a fraction of 1.

    Returns:
        float: The depreciation for the asset.

    .. Note::
        The function takes as input the cost of the asset, the salvage_value of the asset at the end of its useful
        life, the useful_life of the asset in years, and the rate of depreciation as a fraction of 1. It computes the
        depreciation for the asset using the declining balance method, which assumes that the asset depreciates by a
        fixed percentage of its remaining book value each year.

    Examples:
    ---------
    Suppose a company purchases a delivery truck for $50,000, with an expected salvage value of $5,000 after 5 years.
    The company expects to use the truck for 5 years. The depreciation rate for the truck is 30% per year using the
    declining balance method. We can compute the depreciation for the truck using the
    decline function::
        depreciation = syse.decline(cost=50000, salvage_value=5000, useful_life=5, rate=0.3)
        print(f"The annual depreciation for the truck is ${depreciation/5:.2f}.")
        Output = The annual depreciation for the truck is $10717.15.
    This means that the company can deduct $10,717.15 each year for 5 years as a depreciation expense for tax purposes.
    At the end of 5 years, the book value of the truck will be $5,000, which is the expected salvage value.
    """
    accumulated_depreciation = 0.0
    book_value = cost

    for year in range(1, useful_life + 1):
        depreciation = book_value * rate
        if book_value - depreciation < salvage_value:
            depreciation = book_value - salvage_value
        accumulated_depreciation += depreciation
        book_value -= depreciation

    return accumulated_depreciation


# Double Declining Balance Method:

def double(cost: float, salvage_value: float, useful_life: int, rate: float) -> float:
    """
    Compute the depreciation for an asset using the double declining balance method.

    Parameters:
        cost (float): The cost of the asset.
        salvage_value (float): The salvage value of the asset.
        useful_life (int): The useful life of the asset.
        rate (float): The depreciation rate, expressed as a fraction of 1.

    Returns:
        float: The depreciation for the asset.

    .. Note::
        The function takes the same inputs as the declining_balance_method function, but computes the depreciation
        using the double declining balance method. The double declining balance method assumes that the asset
        depreciates by a fixed percentage of its remaining book value each year, but that the rate of depreciation is
        twice the straight-line rate.

    Examples:
    ---------
    Suppose a company purchases a printing press for $100,000, with an expected salvage value of $10,000 after 4 years.
    The company expects to use the printing press for 4 years. The depreciation rate for the printing press is 50% per
    year using the double declining balance method. We can compute the depreciation for the printing press using the
    double function::
        depreciation = syse.double(cost=100000, salvage_value=10000, useful_life=4, rate=0.5)
        print(f"The total depreciation for the printing press over 4 years is ${depreciation:.2f}.")
        Output = The total depreciation for the printing press over 4 years is $102000.00.
    This means that the company can deduct $102,000 as a depreciation expense over the 4-year life of the printing
    press for tax purposes. At the end of 4 years, the book value of the printing press will be $10,000, which is the
    expected salvage value.
    """
    accumulated_depreciation = 0.0
    book_value = cost

    for year in range(1, useful_life + 1):
        depreciation = book_value * rate * 2
        if book_value - depreciation < salvage_value:
            depreciation = book_value - salvage_value
        accumulated_depreciation += depreciation
        book_value -= depreciation

    return accumulated_depreciation


# Units of Production Method:

def units(cost, salvage, expected_units_lifetime, production_t, t):
    """
    Calculate the units-of-production depreciation and book value at the end of year t.

    Parameters:
        cost (float): The original cost of the asset
        salvage (float): The estimated salvage value of the asset at the end of its useful life
        expected_units_lifetime (float): The total expected number of units produced over the asset's lifetime
        production_t (numpy array): A numpy array of units produced per year for t years
        t (int): Number of years

    Returns:
        float: Depreciation expense in year t
        float: Book value at the end of year t

    Examples:
    ---------
    On January 1, Miners, Inc. bought a backhoe for $500,000. They expect to use it for 5,000 hours, about 1,000 per
    year for 5 years, before selling it for $100,000. What is the depreciation expense for the first year if they opt
    for units-of-production method and used it for 1,500 hours?
    ::
           cost = 500000
           salvage = 100000
           expected_units_lifetime = 5000
           production_t = np.array([1500])  # Only the first year's usage is given, so the array has one element
           t = 1
           # Calculate depreciation expense for the first year
           syse.units(cost, salvage, expected_units_lifetime, production_t, t)
           Output = 120000.0


    """
    # Calculate Depreciation per Unit
    depreciation_per_unit = (cost - salvage) / expected_units_lifetime
    # Calculate Depreciation Expense for each year
    depreciation_expense_t = production_t * depreciation_per_unit
    # Calculate accumulated depreciation
    accumulated_depreciation = np.sum(depreciation_expense_t[:t])
    # Calculate book value at the end of year t
    book_value_t = cost - accumulated_depreciation

    return depreciation_expense_t[t-1]


# Probability & Statistics


# Simple Moving Average

def sma(data, n):
    """
    Calculate the simple moving average of a list of data points for the most recent n periods.

    Parameters:
        data (list): A list of data points.
        n (int): The number of most recent observations to be used.

    Returns:
        float: The simple moving average.

    Examples:
    ---------
    Suppose you are the owner of a retail store, and you want to analyze the sales performance of your store for the
    last 10 days. You have a list of the daily sales figures for the past 10 days. You can use the above Python
    function to calculate the simple moving average of the sales figures over the past 10 days to get an idea of the
    store's overall sales trend. This information can be useful in determining whether you need to adjust your
    inventory or marketing strategies to improve sales.


    """
    if len(data) < n:
        return None
    else:
        return sum(data[-n:]) / n


# Weighted Moving Average

def wma(data, weights):
    """
    Calculate the weighted moving average of a list of data points using a list of weights.

    Parameters:
        data (list): A list of data points.
        weights (list): A list of weights to be applied to each data point.

    Returns:
        float: The weighted moving average.

    Examples:
    ---------
        Suppose you are a financial analyst, and you want to analyze the stock price of a company over the
        past 5 days. You have a list of the daily closing prices for the past 5 days, and you believe that the
        most recent day's price should be given a higher weight than the previous days. You can use the above
        Python function to calculate the weighted moving average of the stock prices over the past 5 days,
        where the weight of the most recent day's price is 0.5 and the weights of the previous days are 0.3, 0.1,
        and 0.1 respectively. This information can be useful in determining whether the stock price is trending up or
        down and making investment decisions accordingly.

    """
    if len(data) != len(weights):
        return None
    else:
        return sum([data[i] * weights[i] for i in range(len(data))]) / sum(weights)


# Normal Distribution
# def norm():
#
#     """
#
#
#     :return:
#
#     Examples
#     --------
#     Suppose the lengths of telephone calls form a normal distribution with a mean length
#     of 8.0 min and a standard deviation of 2.5 min. The probability that a telephone call
#     selected at random will last more than 15.5 min is most nearly:
#
#     **A.** 0.0013\n
#     **B.** 0.0026\n
#     **C.** 0.2600\n
#     **D.** 0.9987
#
#     """


# Simple Linear Regression

def line(x, y):
    """
    Calculate slope and intercept of the linear regression line that best fits the data.

    Parameters:
        x: list of x-values
        y: list of y-values

    Returns:
        tuple: (slope, intercept)

    Notes
    -----

    Examples:
    ---------
    """
    # Calculate the mean of x and y
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    # Calculate the slope
    numerator = 0
    denominator = 0
    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2
        slope = numerator / denominator
    # Calculate the intercept
    intercept = y_mean - (slope * x_mean)
    # Return the slope and intercept
    return (slope, intercept)


# Modeling & Quantitative Analysis


# Maximize - Linear Programming

def linear_pro(costs, budget):
    """
    A Python function for linear programming. (Placeholder for a better function)

    Parameters:
        costs : A list of costs associated with a project.
        budget : The total budget allocated to the project.

    Returns:
        float: A list of the maximum values that can be allocated to each cost while staying within the budget.

    .. Caution::
        This function seeks to **maximize** the values.

    Examples:
    ---------
      ::
        costs = [1000, 2000, 3000]
        budget = 5000
        solution = syse.linear_pro(costs, budget)
        print(solution)
        Output= [1000.0, 2000.0, 1000.0]
    """
    # Initialize the solution list
    # Calculate the total of all costs
    solution = []
    total_costs = sum(costs)

    # Check if the total costs exceed the budget
    # Return an empty list if the budget is exceeded
    if total_costs > budget:
        return solution
    # Iterate through each cost
    # Calculate the maximum value that can be allocated to the cost
    # Append the maximum value to the solution list
    # Return the solution list
    for cost in costs:
        max_value = budget * (cost/total_costs)
        solution.append(max_value)
        return solution


# Engineering Management


# Program Evaluation Review Technique (PERT)

def pert(*tasks):
    """
    Calculate the expected time and standard deviation it will take to realistically finish a project using the PERT
    estimate formula: (O +4M + P)/6

    Parameters:
           tasks: np.ndarray or list of estimated times for each task

    Returns:
           float: expected time & standard deviation
       

    Examples:
    ---------
    You are working on a project to build a new office building.
    You need to estimate the time required to complete the project.
    You have identified the following tasks: **1.** Design the building, **2.** Purchase materials, **3.** Construct the building,
    and **4.** Finish the interior. You can estimate the time required to complete the project using the PERT function::
           # Define the tasks for each stage of the project
           DesignBuilding = [60, 70, 100]
           PurchaseMaterials = [60, 70, 100]
           ConstructBuilding = [60, 70, 100]
           FinishInterior = [60, 70, 100]
           syse.pert(DesignBuilding, PurchaseMaterials, ConstructBuilding, FinishInterior)
           Output = ([73.33333333, 73.33333333, 73.33333333, 73.33333333]),
                    [6.66666667, 6.66666667, 6.66666667, 6.66666667]))
    """
    # Convert input to NumPy arrays
    tasks = [np.array(task) for task in tasks]
    # Extract the optimistic, most likely, and pessimistic times for each set of tasks
    optimistic_time = [task[0] for task in tasks]
    most_likely_time = [task[1] for task in tasks]
    pessimistic_time = [task[2] for task in tasks]
    # Calculate the expected time using the PERT formula
    expected_time = (np.array(optimistic_time) + 4 * np.array(most_likely_time) + np.array(pessimistic_time)) / 6
    # Calculate the standard deviation using the PERT formula
    standard_deviation = (np.array(pessimistic_time) - np.array(optimistic_time)) / 6
    # Return the expected time and standard deviation
    return expected_time, standard_deviation


# Network Analysis

# def network():
#     """
#
#     :return:
#
#     Examples
#     --------
#     The project network below shows activity durations in weeks. There is a likelihood
#     that Activity D cannot start until the end of Week 3. In this context, the earliest that
#     the project can be completed is the end of Week:
#
#     **A.** 15\n
#     **B.** 17\n
#     **C.** 18\n
#     **D.** 43
#     """

# Process Analysis

# def process():
#     """
#
#     :return:
#
#     Examples
#     --------
#     The following process creates 10,000 good units per year.
#     The scrap rate of Process C is 20%. Process B has a scrap rate of 10%. To ensure that the
#     raw material input to the system at Process A is limited to 18,519 or less, the maximum
#     allowable scrap rate for Process A is most nearly:
#
#     **A.** 10%\n
#     **B.** 20%\n
#     **C.** 25%\n
#     **D.** 75%
#     """


# Economic Order Quantity (EOQ)

def eoq(a: float, d: float, h: float) -> float:
    """
    Calculate the economic order quantity (EOQ) for an instantaneous replenishment inventory model.

    Parameters:
        A (float): The cost to place one order.
        D (float): The number of units used per year.
        h (float): The holding cost per unit per year.

    Returns:
        float: The EOQ that minimizes the total annual inventory cost.

    Examples:
    ---------
    Let's say that a small business that sells specialty coffee beans uses an instantaneous replenishment
    inventory model to manage its inventory of beans. The business purchases its coffee beans from a supplier and
    pays a fixed cost of $50 to place an order, regardless of the quantity ordered. The business uses 500 bags of
    coffee beans per year, and the holding cost per bag per year is $5.
    To determine the optimal order quantity, we can use the eoq function::
        A = 50
        D = 500
        h = 5
        eoq = syse.eoq(A, D, h)
        Output = 100
    """
    eoq = math.sqrt((2 * a * d) / h)
    return eoq


# Economic Manufacturing Quantity (EMQ)

def emq(a: float, d: float, h: float, r: float) -> float:
    """
    Calculate the economic manufacturing quantity (EMQ) for a finite replenishment rate inventory model.

    Parameters:
        A (float): The cost to place one order.
        D (float): The number of units used per year.
        h (float): The holding cost per unit per year.
        R (float): The replenishment rate.

    Returns:
        float: The EMQ that minimizes the total annual inventory cost.

    .. Note::
        The EMQ formula assumes the same conditions as the EOQ formula
        (i.e., constant and known demand, no stockouts, constant and known ordering
        costs and holding costs), but also assumes that the replenishment rate is
        finite. The EMQ represents the optimal production quantity that minimizes
        the total annual inventory cost, including both holding costs and ordering
        costs, when production is constrained by a finite rate of replenishment.
        Note that the formula assumes that the replenishment rate is given in
        units per day, and that the annual demand is normalized to units per
        day by dividing by 365.

    Examples:
    ---------
    Let's say that a manufacturer produces widgets using a finite replenishment rate inventory model to manage its
    inventory of raw materials. The manufacturer purchases a raw material from a supplier and pays a fixed cost of
    $100 to place an order, regardless of the quantity ordered. The manufacturer uses 10,000 units of the raw material
    per year, and the holding cost per unit per year is $8. The supplier has a limited production capacity and can only
    replenish the manufacturer's inventory at a maximum rate of 500 units per day.
    To determine the optimal production quantity, we can use the emq function::
        A = 100
        D = 10000
        h = 8
        R = 500
        emq = syse.emq(A, D, h, R)
        print(emq)
        Output = 514.29

    .. Important::
        Again, note that this example is simplified and does not take into account other factors that could
        influence the manufacturer's decision-making, such as variability in demand and lead times, stockout costs, and
        other costs associated with ordering and holding inventory. Nonetheless, the EMQ model provides a useful starting
        point for inventory management decisions in a constrained production environment.
    """
    emq = math.sqrt((2 * a * d) / (h * (1 - (d / (r * 365)))))
    return emq
