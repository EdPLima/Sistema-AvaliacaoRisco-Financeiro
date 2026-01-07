"""
Sistema de Avaliação de Risco Financeiro - Interface Streamlit

Este módulo implementa a interface web para avaliação de risco de crédito,
permitindo que usuários insiram dados do cliente e recebam uma avaliação
com probabilidade de inadimplência, nível de confiança e classificação de risco.

Arquitetura:
- Interface de entrada: Formulário com campos para todas as features necessárias
- Processamento: Comunicação com API FastAPI para inferência do modelo
- Interface de saída: Exibição formatada dos resultados com métricas e classificação visual
"""

import os
import streamlit as st
import requests
import pandas as pd
import json
from typing import Any, Dict, Optional
from datetime import datetime


# ============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ============================================================================

# URL padrão da API - pode ser sobrescrita por variável de ambiente
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Threshold padrão para classificação de risco
DEFAULT_THRESHOLD = 0.42

# Configuração de níveis de risco baseado em probabilidade
RISCO_BAIXO_MAX = 0.30      # Probabilidade <= 30%: Risco Baixo
RISCO_MEDIO_MAX = 0.60      # Probabilidade <= 60%: Risco Médio
# Probabilidade > 60%: Risco Alto

# Valores permitidos para campos categóricos
OPCOES_HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "OTHER"]
OPCOES_LOAN_INTENT = [
    "EDUCATION", 
    "MEDICAL", 
    "PERSONAL", 
    "VENTURE", 
    "HOMEIMPROVEMENT", 
    "DEBTCONSOLIDATION"
]
OPCOES_LOAN_GRADE = ["A", "B", "C", "D", "E", "F", "G"]
OPCOES_DEFAULT_HISTORY = ["N", "Y"]
OPCOES_FAIXA_ETARIA = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def calcular_nivel_risco(probabilidade: float) -> str:
    """
    Calcula o nível de risco baseado na probabilidade de inadimplência.
    
    Args:
        probabilidade: Probabilidade de inadimplência (0.0 a 1.0)
    
    Returns:
        String com o nível de risco: "Baixo", "Médio" ou "Alto"
    """
    if probabilidade <= RISCO_BAIXO_MAX:
        return "Baixo"
    elif probabilidade <= RISCO_MEDIO_MAX:
        return "Médio"
    else:
        return "Alto"


def calcular_nivel_confianca(probabilidade: float, threshold: float) -> float:
    """
    Calcula o nível de confiança da predição baseado na distância da probabilidade
    em relação ao threshold de decisão.
    
    A confiança é maior quando a probabilidade está mais distante do threshold,
    indicando maior certeza na classificação.
    
    Args:
        probabilidade: Probabilidade de inadimplência (0.0 a 1.0)
        threshold: Threshold de decisão usado pelo modelo
    
    Returns:
        Nível de confiança normalizado entre 0.0 e 1.0
    """
    distancia = abs(probabilidade - threshold)
    # Normaliza considerando que a distância máxima possível é 1.0
    confianca = min(distancia * 2, 1.0)
    return confianca


def obter_cor_risco(nivel_risco: str) -> str:
    """
    Retorna a cor correspondente ao nível de risco para uso em visualizações.
    
    Args:
        nivel_risco: Nível de risco ("Baixo", "Médio" ou "Alto")
    
    Returns:
        Código hexadecimal da cor
    """
    cores = {
        "Baixo": "#28a745",    # Verde
        "Médio": "#ffc107",     # Amarelo/Laranja
        "Alto": "#dc3545"        # Vermelho
    }
    return cores.get(nivel_risco, "#6c757d")  # Cinza como padrão


def chamar_api_predicao(api_url: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza chamada HTTP POST para o endpoint de predição da API.
    O threshold é fixo em 0.42 (threshold ideal calculado pelo modelo).
    
    Args:
        api_url: URL base da API
        features: Dicionário com as features do cliente
    
    Returns:
        Dicionário com os resultados da predição
    
    Raises:
        requests.HTTPError: Se a requisição falhar
        requests.RequestException: Se houver erro de conexão
    """
    url = api_url.rstrip("/") + "/predict"
    payload = {"features": features, "threshold": DEFAULT_THRESHOLD}
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao conectar com a API: {e}")
        raise


# ============================================================================
# FUNÇÕES DE INTERFACE
# ============================================================================

def renderizar_sidebar():
    """
    Renderiza a barra lateral com configurações da aplicação.
    Permite ao usuário configurar a URL da API.
    O threshold é fixo em 0.42 (threshold ideal calculado pelo modelo).
    """
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Configuração da API
        api_url = st.text_input(
            "URL da API", 
            value=DEFAULT_API_URL,
            help="URL do servidor da API de predição"
        )
        
        st.markdown("---")
        
        # Informações sobre threshold e níveis de risco
        st.subheader("📊 Threshold de Decisão")
        st.info(f"**Threshold fixo: {DEFAULT_THRESHOLD}**\n\nEste valor foi calculado otimizando a estatística KS (Kolmogorov-Smirnov) durante o treinamento do modelo.")
        
        st.subheader("📊 Níveis de Risco")
        st.markdown("""
        - **Baixo**: Probabilidade ≤ 30%
        - **Médio**: Probabilidade ≤ 60%
        - **Alto**: Probabilidade > 60%
        """)
        
        st.markdown("---")
        
        # Status da conexão
        if st.button("🔍 Verificar Conexão"):
            try:
                health_url = api_url.rstrip("/") + "/health"
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    st.success("✅ API conectada com sucesso!")
                else:
                    st.warning(f"⚠️ API retornou status {resp.status_code}")
            except Exception as e:
                st.error(f"❌ Erro ao conectar: {e}")
    
    return api_url


def renderizar_formulario_entrada() -> Optional[Dict[str, Any]]:
    """
    Renderiza o formulário de entrada de dados do cliente.
    Organiza os campos em seções lógicas para melhor UX.
    
    Returns:
        Dicionário com as features do cliente ou None se o formulário não foi submetido
    """
    st.header("📝 Dados do Cliente")
    st.markdown("Preencha os dados abaixo para realizar a avaliação de risco.")
    
    # Container principal do formulário
    with st.form("formulario_avaliacao", clear_on_submit=False):
        
        # Seção 1: Informações Pessoais
        st.subheader("👤 Informações Pessoais")
        col1, col2 = st.columns(2)
        
        with col1:
            person_income = st.number_input(
                "Renda Anual (R$)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                help="Renda anual do cliente em reais"
            )
            
            person_home_ownership = st.selectbox(
                "Tipo de Residência",
                options=OPCOES_HOME_OWNERSHIP,
                index=0,
                help="Situação de moradia do cliente"
            )
        
        with col2:
            person_emp_length = st.number_input(
                "Tempo de Emprego (anos)",
                min_value=0.0,
                value=5.0,
                step=0.5,
                help="Tempo de experiência profissional em anos"
            )
            
            cb_person_cred_hist_length = st.number_input(
                "Histórico de Crédito (anos)",
                min_value=0,
                value=3,
                step=1,
                help="Tempo de histórico de crédito em anos"
            )
        
        # Seção 2: Informações do Empréstimo
        st.subheader("💰 Informações do Empréstimo")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_amnt = st.number_input(
                "Valor do Empréstimo (R$)",
                min_value=0.0,
                value=10000.0,
                step=1000.0,
                help="Valor total solicitado"
            )
            
            loan_int_rate = st.number_input(
                "Taxa de Juros (%)",
                min_value=0.0,
                max_value=100.0,
                value=12.0,
                step=0.1,
                help="Taxa de juros anual do empréstimo"
            )
        
        with col2:
            loan_intent = st.selectbox(
                "Finalidade do Empréstimo",
                options=OPCOES_LOAN_INTENT,
                index=0,
                help="Motivo da solicitação do empréstimo"
            )
            
            loan_grade = st.selectbox(
                "Grau de Risco",
                options=OPCOES_LOAN_GRADE,
                index=2,
                help="Classificação de risco atribuída ao empréstimo"
            )
        
        with col3:
            loan_percent_income = st.number_input(
                "Percentual da Renda (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=0.1,
                help="Percentual da renda comprometida com o empréstimo"
            )
        
        # Seção 3: Histórico e Demografia
        st.subheader("📋 Histórico e Demografia")
        col1, col2 = st.columns(2)
        
        with col1:
            cb_person_default_on_file = st.selectbox(
                "Histórico de Inadimplência",
                options=OPCOES_DEFAULT_HISTORY,
                index=0,
                help="Se o cliente já teve inadimplência anterior (Y=Sim, N=Não)"
            )
        
        with col2:
            faixa_etaria = st.selectbox(
                "Faixa Etária",
                options=OPCOES_FAIXA_ETARIA,
                index=0,
                help="Faixa etária do cliente"
            )
        
        # Botão de submissão
        submitted = st.form_submit_button(
            "🔍 Avaliar Risco",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            # Monta o dicionário de features no formato esperado pela API
            features = {
                "person_income": float(person_income),
                "person_home_ownership": person_home_ownership,
                "person_emp_length": float(person_emp_length),
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": float(loan_amnt),
                "loan_int_rate": float(loan_int_rate),
                "loan_percent_income": float(loan_percent_income) / 100.0,  # Converte % para decimal
                "cb_person_default_on_file": cb_person_default_on_file,
                "cb_person_cred_hist_length": int(cb_person_cred_hist_length),
                "faixa_etaria": faixa_etaria
            }
            
            return features
    
    return None


def renderizar_resultados(resultado_api: Dict[str, Any], threshold: float):
    """
    Renderiza os resultados da avaliação de risco de forma visual e formatada.
    
    Args:
        resultado_api: Dicionário retornado pela API com os resultados
        threshold: Threshold usado na classificação
    """
    st.header("📊 Resultados da Avaliação")
    
    # Extrai valores do resultado da API
    prob_default = resultado_api.get("probabilidade_default", 0.0)
    
    # Usa valores da API se disponíveis, caso contrário calcula localmente
    prob_percent = resultado_api.get("probabilidade_percentual", prob_default * 100.0)
    nivel_risco = resultado_api.get("nivel_risco", calcular_nivel_risco(prob_default))
    nivel_confianca = resultado_api.get("nivel_confianca", calcular_nivel_confianca(prob_default, threshold))
    cor_risco = obter_cor_risco(nivel_risco)
    
    # Container principal de resultados
    with st.container():
        # Métricas principais em cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Probabilidade de Inadimplência",
                value=f"{prob_percent:.2f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Nível de Confiança",
                value=f"{nivel_confianca * 100:.2f}%",
                delta=None
            )
        
        with col3:
            # Card customizado para nível de risco com cor
            st.markdown(f"""
            <div style="
                background-color: {cor_risco}20;
                border-left: 4px solid {cor_risco};
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            ">
                <h3 style="margin: 0; color: {cor_risco};">
                    Nível de Risco: {nivel_risco}
                </h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Seção de detalhes
        st.subheader("📋 Detalhes da Avaliação")
        
        # Tabela com informações detalhadas
        detalhes_data = {
            "Métrica": [
                "Probabilidade de Não Pagar",
                "Nível de Risco",
                "Nível de Confiança",
                "Threshold Utilizado",
                "Classificação Binária"
            ],
            "Valor": [
                f"{prob_percent:.2f}%",
                nivel_risco,
                f"{nivel_confianca * 100:.2f}%",
                f"{threshold:.2f}",
                resultado_api.get("classificacao", "N/A")
            ]
        }
        
        df_detalhes = pd.DataFrame(detalhes_data)
        st.dataframe(df_detalhes, use_container_width=True, hide_index=True)
        
        # Barra de progresso visual para probabilidade
        st.markdown("### 📈 Visualização da Probabilidade")
        st.progress(prob_default, text=f"Probabilidade de Inadimplência: {prob_percent:.2f}%")
        
        # Exibe probabilidade formatada corretamente
        st.markdown(f"**Probabilidade de não pagar: {prob_percent:.2f}%**")
        
        # Interpretação do resultado
        st.markdown("---")
        st.subheader("💡 Interpretação")
        
        if nivel_risco == "Baixo":
            st.success(f"""
            **Risco Baixo**: A probabilidade de inadimplência é de {prob_percent:.2f}%, 
            indicando um perfil de baixo risco. O cliente apresenta características 
            favoráveis para aprovação do empréstimo.
            """)
        elif nivel_risco == "Médio":
            st.warning(f"""
            **Risco Médio**: A probabilidade de inadimplência é de {prob_percent:.2f}%, 
            indicando um perfil de risco moderado. Recomenda-se análise adicional 
            e possível solicitação de garantias adicionais.
            """)
        else:
            st.error(f"""
            **Risco Alto**: A probabilidade de inadimplência é de {prob_percent:.2f}%, 
            indicando um perfil de alto risco. Recomenda-se rejeição da solicitação 
            ou análise criteriosa com condições especiais.
            """)
        
        # Informações técnicas (expansível)
        with st.expander("🔧 Informações Técnicas"):
            st.json(resultado_api)


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal da aplicação Streamlit.
    Orquestra o fluxo completo: configuração, entrada de dados, processamento e exibição de resultados.
    """
    # Configuração da página
    st.set_page_config(
        page_title="Sistema de Avaliação de Risco Financeiro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título e descrição principal
    st.title("📊 Sistema de Avaliação de Risco Financeiro")
    st.markdown("""
    Sistema inteligente para avaliação de risco de crédito baseado em Machine Learning.
    Insira os dados do cliente abaixo para obter uma avaliação completa do risco de inadimplência.
    """)
    
    st.markdown("---")
    
    # Renderiza sidebar e obtém configurações
    api_url = renderizar_sidebar()
    
    # Renderiza formulário de entrada
    features = renderizar_formulario_entrada()
    
    # Processa predição se o formulário foi submetido
    if features is not None:
        try:
            with st.spinner("🔄 Processando avaliação..."):
                resultado = chamar_api_predicao(api_url, features)
            
            st.markdown("---")
            renderizar_resultados(resultado, DEFAULT_THRESHOLD)
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erro ao processar a avaliação: {e}")
            st.info("Verifique se a API está rodando e se a URL está correta nas configurações.")
        except Exception as e:
            st.error(f"❌ Erro inesperado: {e}")
            st.exception(e)


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
