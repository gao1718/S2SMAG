"""经典三法在明确历史段内估计：不改原数据、不裁参数、不做末点平移。"""
import numpy as np


def geometric(beta, steps):
    if beta == 1.:
        return steps.astype(float), np.ones(len(steps))
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        if beta > 0:
            power=np.exp(steps*np.log(beta))
            total=np.expm1(steps*np.log(beta))/np.expm1(np.log(beta))
        else:
            power=np.power(beta,steps)
            total=(power-1)/(beta-1)
    return total,power


def fit_segment(method,t_prefix,s_prefix,t_future,start_index,*,min_history=7,asaoka_interval=1):
    """future仅时间；参数/取点只接当前prefix。ThreePoint是真正三点解析式。"""
    t,s,future=(np.asarray(value,dtype=np.float64).reshape(-1) for value in (t_prefix,s_prefix,t_future))
    if len(t)!=len(s) or len(t)<3 or not all(np.isfinite(a).all() for a in (t,s,future)):
        raise ValueError("历史/未来时间或原始沉降不合法。")
    if not np.allclose(np.diff(t),1.,rtol=0,atol=1e-8):
        raise ValueError("本次C1协议为实际逐日观测；不自行插值或改日期。")
    if len(future) and np.any(future<=t[-1]): raise ValueError("预测时间必须晚于末观测。")
    indices=np.arange(max(0,int(start_index)),len(t))
    segment_t,segment_s=t[indices],s[indices]
    result={"prediction":np.full(len(future),np.nan),"historical_prediction":np.full(len(indices),np.nan),
            "historical_recursive_prediction":np.full(len(indices),np.nan),"history_indices":indices,
            "used_indices":np.array([],dtype=int),"parameters":{},"status":"inapplicable","reason":"",
            "diagnostics":{"method":method,"available_prefix_n":len(t),"history_n":len(indices),
                           "history_first_index":int(indices[0]) if len(indices) else None,
                           "history_last_index":len(t)-1,"future_truth_used":False,"coefficient_clipped":False,
                           "last_observation_residual_shift":False,"observation_modified":False}}
    if len(indices)<min_history:
        result["reason"]=f"指定历史段仅{len(indices)}点，不满足预先固定至少{min_history}点；未补点或另换有利起点。"
        return result
    warnings=[]
    diag=result["diagnostics"]
    if method=="Hyperbolic":
        tau=segment_t-segment_t[0]
        slope=float(np.linalg.lstsq(np.column_stack((np.ones(len(indices)),tau)),segment_s,rcond=None)[0][1])
        direction=-1. if slope<0 else 1.
        u=direction*(segment_s-segment_s[0])
        if np.any(u[1:]<=0):
            result["reason"]="相对沉降增量出现非正值，时沉比无法全部计算；未删除观测。";return result
        design=np.column_stack((np.ones(len(indices)-1),tau[1:]))
        response=tau[1:]/u[1:]
        coefficients,_,rank,_=np.linalg.lstsq(design,response,rcond=None)
        a,b=map(float,coefficients)
        evaluate=lambda query: segment_s[0]+direction*(query-segment_t[0])/(a+b*(query-segment_t[0]))
        with np.errstate(divide="ignore",over="ignore",invalid="ignore"):
            prediction=evaluate(future);historical=evaluate(segment_t)
        result["used_indices"]=indices.copy()
        result["parameters"]={"reference_time":float(segment_t[0]),"reference_settlement":float(segment_s[0]),
                              "direction":direction,"a":a,"b":b,"ultimate_settlement":float(segment_s[0]+direction/b) if b else None}
        positive=bool(a>0 and b>0)
        if not positive:warnings.append("经典时沉比OLS不满足a>0,b>0；原参数与有限曲线保留，未夹系数")
        diag.update(estimator="classical_tau_over_u_on_tau_OLS",formula="S0+d*tau/(a+b*tau)",
                    reference_point_excluded_from_ratio="tau=0,u=0 gives0/0; S0 still defines every relative increment",
                    regression_pairs=len(indices)-1,fit_n=len(indices),rank=int(rank),parameter_condition_met=positive,
                    historical_metric_kind="curve_backfit_over_all_available_calibration_stage_points",
                    transformed_RMSE=float(np.sqrt(np.mean((design@coefficients-response)**2))),
                    transformed_tau=tau[1:].tolist(),transformed_tau_over_u=response.tolist())
    elif method=="ThreePoint":
        interval=(len(indices)-1)//2
        chosen=np.array([indices[-1]-2*interval,indices[-1]-interval,indices[-1]])
        selected_t,selected_s=t[chosen],s[chosen]
        dt=float(selected_t[1]-selected_t[0])
        if dt<=0 or not np.isclose(selected_t[2]-selected_t[1],dt):
            result["reason"]="三点实际时间不等距。";return result
        delta1,delta2=np.diff(selected_s)
        if delta1==0:
            result["reason"]="第一增量为零，无法识别有限指数比率。";return result
        ratio=float(delta2/delta1)
        result["used_indices"]=chosen
        result["parameters"]={"selected_indices":chosen.tolist(),"selected_times":selected_t.tolist(),
                              "selected_settlement":selected_s.tolist(),"interval_days":dt,"ratio":ratio}
        if ratio<=0:
            result["reason"]="三点增量比<=0，不存在正实指数参数；未换点。";return result
        if abs(ratio-1)<1e-8:
            evaluate=lambda query:selected_s[-1]+delta2/dt*(query-selected_t[-1])
            result["parameters"].update(ultimate_settlement=None,decay_k_per_day=0.,linear_limit=True)
            warnings.append("增量比近1，明确使用线性极限，无可识别有限最终沉降")
        else:
            ultimate=float(selected_s[0]+delta1/(1-ratio))
            decay=float(-np.log(ratio)/dt)
            evaluate=lambda query:ultimate+(selected_s[-1]-ultimate)*np.exp(-decay*(query-selected_t[-1]))
            result["parameters"].update(ultimate_settlement=ultimate,decay_k_per_day=decay,linear_limit=False)
        with np.errstate(over="ignore",invalid="ignore"):
            prediction=evaluate(future);historical=evaluate(segment_t)
        if ratio>=1:warnings.append("r>=1不满足收敛三点法假设；解析参数及有限外推保留为警告")
        diag.update(estimator="TRUE_classical_equal_time_three_point_analytic",fit_n=3,interval_days=dt,
                    selected_indices=chosen.tolist(),parameter_condition_met=bool(0<ratio<1),
                    historical_metric_kind="backfit_over_calibration_stage; estimator_uses_exactly3actual_points")
    elif method=="Asaoka":
        delta=int(asaoka_interval)
        if delta<1: raise ValueError("Asaoka间隔须为正整数天。")
        sampled=indices[-1]-np.arange((indices[-1]-indices[0])//delta,-1,-1)*delta
        if len(sampled)<3:
            result["reason"]="指定等日间隔且包含末观测后不足3个样本，无法识别相邻OLS；未补点。";return result
        relative=(future-t[-1])/delta
        steps=np.rint(relative).astype(np.int64)
        if not np.allclose(relative,steps,atol=1e-8,rtol=0):
            result["reason"]="诊断未来不在该Asaoka原生离散网格，未把多日系数误当1日。";return result
        selected=s[sampled]
        design=np.column_stack((np.ones(len(sampled)-1),selected[:-1]))
        coefficients,_,rank,_=np.linalg.lstsq(design,selected[1:],rcond=None)
        alpha,beta=map(float,coefficients)
        total,power=geometric(beta,steps)
        with np.errstate(over="ignore",invalid="ignore"):
            prediction=selected[-1]*power+alpha*total
        historical=np.full(len(indices),np.nan)
        historical[sampled[1:]-indices[0]]=design@coefficients
        result["used_indices"]=sampled
        ht,hp=geometric(beta,np.arange(len(sampled)))
        recursive=np.full(len(indices),np.nan);recursive[sampled-indices[0]]=selected[0]*hp+alpha*ht
        result["historical_recursive_prediction"]=recursive
        result["parameters"]={"alpha_raw_signed":alpha,"beta":beta,"interval_days":delta,
                              "forecast_anchor_time":float(t[-1]),"forecast_anchor_settlement":float(s[-1]),
                              "sampled_indices":sampled.tolist(),"ultimate_settlement":alpha/(1-beta) if beta!=1 else None}
        physical=bool(0<=beta<1)
        if not physical:warnings.append("beta不在[0,1)，不满足单调收敛条件；原系数及有限发散/振荡预测保留")
        if len(sampled)<7:warnings.append("多日间隔仅有少量实际样本，作为弱识别敏感性诊断保留")
        diag.update(estimator="rawsigned_equal_interval_adjacent_OLS",fit_n=len(sampled),regression_pairs=len(sampled)-1,
                    interval_days=delta,rank=int(rank),last_observation_included=True,
                    parameter_condition_met=physical,mathematically_convergent=bool(abs(beta)<1),
                    historical_metric_kind="one_step_regression_residuals_on_actual_equal_interval_pairs")
    else:
        raise ValueError(method)
    finite=np.isfinite(prediction)
    if not finite.all():
        prediction=np.asarray(prediction).copy();prediction[~finite]=np.nan
        warnings.append("只有溢出位置记NaN，有限极大值保留")
    residual=historical-segment_s;valid_history=np.isfinite(residual)
    diag.update(historical_MAE=float(np.abs(residual[valid_history]).mean()) if valid_history.any() else None,
                historical_RMSE=float(np.sqrt(np.mean(residual[valid_history]**2))) if valid_history.any() else None,
                last_historical_residual=float(residual[-1]) if np.isfinite(residual[-1]) else None,
                finite_future_count=int(finite.sum()),future_count=len(future),parameter_clipping=False)
    result.update(prediction=prediction,historical_prediction=historical,
                  status="failed" if not finite.all() else ("warning" if warnings else "ok"),reason="；".join(warnings))
    return result
