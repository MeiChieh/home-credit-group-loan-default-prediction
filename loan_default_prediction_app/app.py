from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, Optional
import joblib
from hepler import *
import pandas as pd
import numpy as np

model = joblib.load("./vote_model.pkl")


class LoanDefaultPredictionApp(BaseModel):

    flag_own_car: Optional[Literal["N", "Y"]]
    amt_credit: Optional[float]
    amt_annuity: Optional[float]
    amt_goods_price: Optional[float]
    name_income_type: Optional[Literal[
        "Businessman",
        "Commercial associate",
        "Maternity leave",
        "Pensioner",
        "State servant",
        "Student",
        "Unemployed",
        "Working",
    ]]
    name_education_type: Optional[Literal[
        "Academic degree",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Secondary / secondary special",
    ]]
    name_family_status: Optional[Literal[
        "Civil marriage",
        "Married",
        "Separated",
        "Single / not married",
        "Unknown",
        "Widow",
    ]]
    days_birth: Optional[int]
    days_employed: Optional[int]
    days_registration: Optional[float]
    own_car_age: Optional[float]
    occupation_type: Optional[Literal[
        "Accountants",
        "Cleaning staff",
        "Cooking staff",
        "Core staff",
        "Drivers",
        "HR staff",
        "High skill tech staff",
        "IT staff",
        "Laborers",
        "Low-skill Laborers",
        "Managers",
        "Medicine staff",
        "Private service staff",
        "Realty agents",
        "Sales staff",
        "Secretaries",
        "Security staff",
        "Waiters/barmen staff",
    ]]

    region_rating_client: Optional[int]
    reg_city_not_live_city: Optional[int]
    organization_type: Optional[Literal[
        "Advertising",
        "Agriculture",
        "Bank",
        "Business Entity Type 1",
        "Business Entity Type 2",
        "Business Entity Type 3",
        "Cleaning",
        "Construction",
        "Culture",
        "Electricity",
        "Emergency",
        "Government",
        "Hotel",
        "Housing",
        "Industry: type 1",
        "Industry: type 10",
        "Industry: type 11",
        "Industry: type 12",
        "Industry: type 13",
        "Industry: type 2",
        "Industry: type 3",
        "Industry: type 4",
        "Industry: type 5",
        "Industry: type 6",
        "Industry: type 7",
        "Industry: type 8",
        "Industry: type 9",
        "Insurance",
        "Kindergarten",
        "Legal Services",
        "Medicine",
        "Military",
        "Mobile",
        "Other",
        "Police",
        "Postal",
        "Realtor",
        "Religion",
        "Restaurant",
        "School",
        "Security",
        "Security Ministries",
        "Self-employed",
        "Services",
        "Telecom",
        "Trade: type 1",
        "Trade: type 2",
        "Trade: type 3",
        "Trade: type 4",
        "Trade: type 5",
        "Trade: type 6",
        "Trade: type 7",
        "Transport: type 1",
        "Transport: type 2",
        "Transport: type 3",
        "Transport: type 4",
        "University",
        "XNA",
    ]]
    ext_source_1: Optional[float]
    ext_source_2: Optional[float]
    ext_source_3: Optional[float]
    years_build_avg: Optional[float]
    entrances_avg: Optional[float]
    apartments_mode: Optional[float]
    years_build_medi: Optional[float]
    totalarea_mode: Optional[float]
    def_30_cnt_social_circle: Optional[float]
    def_60_cnt_social_circle: Optional[float]
    days_last_phone_change: Optional[float]
    flag_document_3: Optional[int]
    consumer_loans_amt_annuity: Optional[float]
    consumer_loans_amt_application: Optional[float]
    consumer_loans_amt_down_payment: Optional[float]
    consumer_loans_rate_down_payment: Optional[float]
    consumer_loans_name_client_type: Optional[float]
    consumer_loans_name_goods_category: Optional[Literal[
        "Additional Service",
        "Animals",
        "Audio/Video",
        "Auto Accessories",
        "Clothing and Accessories",
        "Computers",
        "Construction Materials",
        "Consumer Electronics",
        "Direct Sales",
        "Education",
        "Fitness",
        "Furniture",
        "Gardening",
        "Homewares",
        "Insurance",
        "Jewelry",
        "Medical Supplies",
        "Medicine",
        "Mobile",
        "Office Appliances",
        "Other",
        "Photo / Cinema Equipment",
        "Sport and Leisure",
        "Tourism",
        "Vehicles",
        "Weapon",
    ]]
    cash_loans_sellerplace_area: Optional[float]
    cash_loans_cnt_payment: Optional[float]
    consumer_loans_cnt_payment: Optional[float]
    cash_loans_name_yield_group: Optional[float]
    cash_loans_product_combination: Optional[Literal[
        "Cash",
        "Cash Street: high",
        "Cash Street: low",
        "Cash Street: middle",
        "Cash X-Sell: high",
        "Cash X-Sell: low",
        "Cash X-Sell: middle",
    ]]
    cash_loans_days_first_due: Optional[float]
    cash_loans_days_last_due_1st_version: Optional[float]
    consumer_loans_days_last_due_1st_version: Optional[float]
    cash_loans_sk_dpd_mean: Optional[float]
    cash_loans_sk_dpd_def_sum: Optional[float]
    cash_loans_sk_dpd_def_mean: Optional[float]
    consumer_loans_sk_dpd_def_mean: Optional[float]
    cash_loans_cnt_inst_org: Optional[float]
    consumer_loans_contract_status: Optional[float]
    cash_loans_cnt_inst_decreases: Optional[Literal[False, True]]
    revolving_loans_atm_draw_ratio_avg: Optional[float]
    revolving_loans_total_monthly_draw_avg: Optional[float]
    revolving_loans_total_still_owes: Optional[float]
    revolving_loans_total_still_owes_int_princ_ratio: Optional[float]
    revolving_loans_interest_rate: Optional[float]
    revolving_loans_pay_owe_ratio_total: Optional[float]
    revolving_loans_sk_dpd_mean_cc_df: Optional[float]
    cash_loans_delay_inst_ratio: Optional[float]
    consumer_loans_delay_inst_ratio: Optional[float]
    revolving_loans_delay_inst_ratio: Optional[float]
    cash_loans_delay_total_days: Optional[float]
    consumer_loans_delay_total_days: Optional[float]
    consumer_loans_total_payment: Optional[float]
    cash_loans_total_pay_owe_diff: Optional[float]
    cash_loans_total_pay_owe_diff_ratio: Optional[float]
    rej_history_count: Optional[float]
    rej_history_reason_mode: Optional[Literal[
        "HC", "LIMIT", "SCO", "SCOFR", "SYSTEM", "VERIF", "XAP", "XNA"
    ]]
    active_credit_count: Optional[float]
    days_credit_active_mean: Optional[float]
    credit_day_overdue_count: Optional[float]
    amt_credit_max_overdue_mean: Optional[float]
    amt_credit_sum_mean: Optional[float]
    amt_credit_sum_debt_mean: Optional[float] = None


class PredictionResult(BaseModel):
    default: int


# Declaring our FastAPI instance
app = FastAPI()

# sample test
sample_data = {
    "flag_own_car": "N",
    "amt_credit": 904500.0,
    "amt_annuity": 38452.5,
    "amt_goods_price": 904500.0,
    "name_income_type": "Working",
    "name_education_type": "Secondary / secondary special",
    "name_family_status": "Married",
    "days_birth": -13228,
    "days_employed": -3282,
    "days_registration": -881.0,
    "own_car_age": 0,
    "occupation_type": "Laborers",
    "region_rating_client": 2,
    "reg_city_not_live_city": 0,
    "organization_type": "Business Entity Type 2",
    "ext_source_1": 0,
    "ext_source_2": 0.6171231269836426,
    "ext_source_3": 0.3312508761882782,
    "years_build_avg": 0,
    "entrances_avg": 0.10339999943971634,
    "apartments_mode": 0.1396999955177307,
    "years_build_medi": 0,
    "totalarea_mode": 0.04650000110268593,
    "def_30_cnt_social_circle": 1.0,
    "def_60_cnt_social_circle": 1.0,
    "days_last_phone_change": -2721.0,
    "flag_document_3": 1,
    "consumer_loans_amt_annuity": 6923.18212890625,
    "consumer_loans_amt_application": 62561.25,
    "consumer_loans_amt_down_payment": 7407.0,
    "consumer_loans_rate_down_payment": 0.13058757781982422,
    "consumer_loans_name_client_type": 2.25,
    "consumer_loans_name_goods_category": "Mobile",
    "cash_loans_sellerplace_area": 0.0,
    "cash_loans_cnt_payment": 18.0,
    "consumer_loans_cnt_payment": 10.0,
    "cash_loans_name_yield_group": 4.0,
    "cash_loans_product_combination": "Cash X-Sell: high",
    "cash_loans_days_first_due": -434.0,
    "cash_loans_days_last_due_1st_version": 76.0,
    "consumer_loans_days_last_due_1st_version": -1102.5,
    "cash_loans_sk_dpd_mean": 0.0,
    "cash_loans_sk_dpd_def_sum": 0.0,
    "cash_loans_sk_dpd_def_mean": 0.0,
    "consumer_loans_sk_dpd_def_mean": 0.0,
    "cash_loans_cnt_inst_org": 18.0,
    "consumer_loans_contract_status": 2.0059523582458496,
    "cash_loans_cnt_inst_decreases": True,
    "revolving_loans_atm_draw_ratio_avg": 0,
    "revolving_loans_total_monthly_draw_avg": 0,
    "revolving_loans_total_still_owes": 0,
    "revolving_loans_total_still_owes_int_princ_ratio": 0,
    "revolving_loans_interest_rate": 0,
    "revolving_loans_pay_owe_ratio_total": 0,
    "revolving_loans_sk_dpd_mean_cc_df": 0,
    "cash_loans_delay_inst_ratio": 0.0,
    "consumer_loans_delay_inst_ratio": 0.17171716690063477,
    "revolving_loans_delay_inst_ratio": 0,
    "cash_loans_delay_total_days": 0.0,
    "consumer_loans_delay_total_days": 9.666666984558105,
    "consumer_loans_total_payment": 81688.484375,
    "cash_loans_total_pay_owe_diff": 0.0,
    "cash_loans_total_pay_owe_diff_ratio": 0.0,
    "rej_history_count": 0,
    "rej_history_reason_mode": "HC",
    "active_credit_count": 4.0,
    "days_credit_active_mean": -1347.6666259765625,
    "credit_day_overdue_count": 6.0,
    "amt_credit_max_overdue_mean": 0.0,
    "amt_credit_sum_mean": 372855.0,
    "amt_credit_sum_debt_mean": 372855.0,
}

float_cols = ["amt_credit",
 "amt_annuity",
 "amt_goods_price",
 "days_registration",
 "own_car_age",
 "ext_source_1",
 "ext_source_2",
 "ext_source_3",
 "years_build_avg",
 "entrances_avg",
 "apartments_mode",
 "years_build_medi",
 "totalarea_mode",
 "def_30_cnt_social_circle",
 "def_60_cnt_social_circle",
 "days_last_phone_change",
 "consumer_loans_amt_annuity",
 "consumer_loans_amt_application",
 "consumer_loans_amt_down_payment",
 "consumer_loans_rate_down_payment",
 "consumer_loans_name_client_type",
 "cash_loans_sellerplace_area",
 "cash_loans_cnt_payment",
 "consumer_loans_cnt_payment",
 "cash_loans_name_yield_group",
 "cash_loans_days_first_due",
 "cash_loans_days_last_due_1st_version",
 "consumer_loans_days_last_due_1st_version",
 "cash_loans_sk_dpd_mean",
 "cash_loans_sk_dpd_def_sum",
 "cash_loans_sk_dpd_def_mean",
 "consumer_loans_sk_dpd_def_mean",
 "cash_loans_cnt_inst_org",
 "consumer_loans_contract_status",
 "revolving_loans_atm_draw_ratio_avg",
 "revolving_loans_total_monthly_draw_avg",
 "revolving_loans_total_still_owes",
 "revolving_loans_total_still_owes_int_princ_ratio",
 "revolving_loans_interest_rate",
 "revolving_loans_pay_owe_ratio_total",
 "revolving_loans_sk_dpd_mean_cc_df",
 "cash_loans_delay_inst_ratio",
 "consumer_loans_delay_inst_ratio",
 "revolving_loans_delay_inst_ratio",
 "cash_loans_delay_total_days",
 "consumer_loans_delay_total_days",
 "consumer_loans_total_payment",
 "cash_loans_total_pay_owe_diff",
 "cash_loans_total_pay_owe_diff_ratio",
 "rej_history_count",
 "active_credit_count",
 "days_credit_active_mean",
 "credit_day_overdue_count",
 "amt_credit_max_overdue_mean",
 "amt_credit_sum_mean",
 "amt_credit_sum_debt_mean"]

int_cols = ["days_birth",
 "days_employed",
 "region_rating_client",
 "reg_city_not_live_city",
 "flag_document_3"]

cate_cols = ["flag_own_car",
 "name_income_type",
 "name_education_type",
 "name_family_status",
 "occupation_type",
 "organization_type",
 "consumer_loans_name_goods_category",
 "cash_loans_product_combination",
 "cash_loans_cnt_inst_decreases",
 "rej_history_reason_mode"]

cate_mode_dict = {
    "flag_own_car": "N",
    "name_income_type": "working",
    "name_education_type": "Secondary / secondary special",
    "name_family_status": "Married",
    "occupation_type": "Laborers",
    "organization_type": "Business Entity Type 3",
    "consumer_loans_name_goods_category": "Mobile",
    "cash_loans_product_combination": "Cash X-Sell: middle",
    "rej_history_reason_mode": "HC",
    "cash_loans_cnt_inst_decreases": "false"
}

  
def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data types of DataFrame columns and handle nulls.

    Parameters:
    - df (DataFrame): The DataFrame to be processed.


    Returns:
    - DataFrame: The DataFrame with converted data types.
    """
    for col in df:
        if col in float_cols:
            df[col] = (
                df[col]
                .apply(lambda x: np.nan if x is None else x)
                .astype("float32")
            )
        elif col in int_cols:
            df[col] = (
                df[col]
                .apply(lambda x: np.nan if x is None else x)
                .astype("int64")
            )
            
            df[col] = df[col]

    return df


# prediction endpoint
@app.post("/predict", response_model=PredictionResult)
def predict(sample_data: LoanDefaultPredictionApp):
    
    try:

        input_data = sample_data.model_dump()
        input_df = pd.DataFrame([input_data])
        input_df = convert_types(input_df)
        print(input_df.dtypes)
        preds = model.predict(input_df)[0]
        result = {"default": preds}

        return result


    except Exception as e:
        print(f"An error occurred: {e}")
        # Raise an HTTP 422 Unprocessable Entity error
        raise HTTPException(status_code=422, detail=str(e))
