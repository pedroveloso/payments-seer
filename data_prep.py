import multiprocessing

import pandas as pd

pd.options.mode.chained_assignment = None # Disabel warnings

cash_position = pd.read_csv("data/home_credit/POS_CASH_balance.csv")

# Function to calculate payment status
def payment_status(days_between_payment_due_date):
    if days_between_payment_due_date < -30:
        return "Advance payment"
    if -30 < days_between_payment_due_date <= 0:
        return "Payment on schedule"
    if 0 < days_between_payment_due_date <= 30:
        return "Late payment within 30 days"
    if 30 < days_between_payment_due_date <= 90:
        return "Late payment 30+ days but no default"
    if days_between_payment_due_date > 90:
        return "Default"



def get_contract_payments_history(sk_id_prev, cash_position_df=cash_position,):
    
    loan_payments = cash_position_df[cash_position_df.SK_ID_PREV == sk_id_prev]
    loan_payments.sort_values(by=['CNT_INSTALMENT_FUTURE'], ascending=False, inplace=True)
    loan_payments = loan_payments[["CNT_INSTALMENT_FUTURE", "SK_DPD"]]
    loan_payments["PAYMENT_STATUS"] = loan_payments["SK_DPD"].apply(lambda x: payment_status(x))
    
    loan_payments = loan_payments[1:]
    
    payments_behavior = [loan_payments.SK_DPD[0:i].tolist() for i in range(0, len(loan_payments.SK_DPD))][1:]

    next_period_status = loan_payments.PAYMENT_STATUS[1:]

    payments_historical_behavior = {
        'payments_history': payments_behavior,
        'next_period_status': next_period_status
    }

    payments_history = pd.DataFrame(payments_historical_behavior)
    
    return payments_history

# Data derivation
completed_contracts = cash_position.query("NAME_CONTRACT_STATUS == 'Completed' | \
                                          NAME_CONTRACT_STATUS == 'Amortized debt' | \
                                          NAME_CONTRACT_STATUS == 'Demand'")['SK_ID_PREV'].unique().tolist()

final_df = pd.DataFrame()
with multiprocessing.Pool() as pool:

    print("Começando multiprocessing =)")
    payment_history = pool.map(get_contract_payments_history, completed_contracts)    
    final_df = final_df.append(payment_history, ignore_index=True)

print(f"Final dataset has {final_df.shape[0]} rows and {final_df.shape[1]} columns")

# Test if everything is right before writing
if len(final_df.next_period_status.unique()) == 1:
    raise ValueError("Seems there is just one class in your data")

final_df.to_parquet('./data/processed/final_df.parquet.gzip', compression="gzip")
