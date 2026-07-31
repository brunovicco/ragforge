# Judge calibration worksheet

> [!WARNING]
> **Legacy evidence — do not label this worksheet.** Run `20260726T185553Z`
> predates exact `judge_contexts` persistence. Its cited-source excerpts are
> not the ordered contexts evaluated by RAGAS, so this artifact cannot produce
> a valid judge-vs-human calibration report. Preserve it for audit only and
> rerun the benchmark before rebuilding the calibration workspace.

Run `20260726T185553Z` · 36 samples · selection seed `regrag-br-calibration-v1`

Score each sample in `labels.json`. The judge's own scores are sealed
in `.judge-sealed.json` and must stay closed until you finish - seeing
them first turns the exercise into agreement-by-anchoring.

## Scale

Use `0.0`, `0.5`, or `1.0`. `_to_ordinal` bins continuous judge scores
into these same three categories, so intermediate values buy nothing.

**Faithfulness** - is every assertion in the answer supported by the
cited authoritative text?

- `1.0` fully supported; no claim goes beyond the cited articles
- `0.5` mostly supported, but at least one claim overreaches, softens an
  exception, drops a condition, or cites an article that does not carry it
- `0.0` a central assertion contradicts the text or has no support

**Answer Relevancy** - does the answer address what was asked?

- `1.0` answers the question directly and completely
- `0.5` partially answers, or answers a related but different question
- `0.0` does not address the question

For an **unanswerable** question, a correct abstention scores `1.0` on
both; a confident answer built from unrelated retrieved text scores `0.0`.

## Watch for (ADR-0018's own list)

Normative negation (`não constitui`, `veda`), exceptions (`salvo`,
`ressalvado`), cross-references (`na forma do art. X`), numerical claims
(deadlines, percentages, thresholds), and partial support - the answer is
true but the cited article is not where it says so. These are exactly the
cases where a general-purpose judge is expected to slip.

---

## 1. `q199-graphrag`

- Class: **exact_factual**
- Strategy: `graphrag`

### Question

O que deve constar do Acordo de Supervisão quanto à confissão do signatário, segundo o art. 101, inciso V da ICVM 607/2019?

### Reference answer (golden set)

Deve constar confissão expressa da participação do signatário do Acordo de Supervisão no ilícito (art. 101, V).

### Cited authoritative text

**`ICVM-607/2019::art-101::inc-v`**

> V – confissão expressa da participação do signatário do Acordo de Supervisão no ilícito;

### Generated answer

Segundo o art. 101, inciso V da Instrução CVM nº 607/2019, o Acordo de Supervisão deve conter a confissão expressa da participação do signatário no ilícito [ICVM-607/2019::art-101::inc-v].

### Your scores

| `q199-graphrag` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 2. `q186-raptor`

- Class: **exact_factual**
- Strategy: `raptor`

### Question

O proponente pode retificar ou desistir da proposta de Acordo de Supervisão, segundo o art. 95 da ICVM 607/2019?

### Reference answer (golden set)

Sim, o proponente poderá retificar ou desistir da proposta a qualquer momento antes da assinatura do respectivo instrumento (art. 95).

### Cited authoritative text

**`ICVM-607/2019::art-95`**

> Art. 95.  O proponente poderá retificar ou desistir da proposta de Acordo de Supervisão a
> qualquer momento antes da assinatura do respectivo instrumento.

### Generated answer

Sim, o proponente pode retificar ou desistir da proposta de Acordo de Supervisão a qualquer momento antes da assinatura do respectivo instrumento [ICVM-607/2019::art-95].

### Your scores

| `q186-raptor` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 3. `q149-contextual`

- Class: **numeric_tabular**
- Strategy: `contextual`

### Question

Qual é um dos valores máximos para a penalidade de multa, segundo o art. 61, inciso I da ICVM 607/2019?

### Reference answer (golden set)

A multa não deverá exceder R$ 50.000.000,00 (cinquenta milhões de reais), entre outros critérios possíveis (art. 61, I).

### Cited authoritative text

**`ICVM-607/2019::art-61::inc-i`**

> I – R$ 50.000.000,00 (cinquenta milhões de reais);

### Generated answer

De acordo com o art. 61, inciso I, da Instrução CVM nº 607/2019, um dos valores máximos para a penalidade de multa é de R$ 50.000.000,00 (cinquenta milhões de reais) [ICVM-607/2019::art-61::inc-i].

### Your scores

| `q149-contextual` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 4. `q149-raptor`

- Class: **numeric_tabular**
- Strategy: `raptor`

### Question

Qual é um dos valores máximos para a penalidade de multa, segundo o art. 61, inciso I da ICVM 607/2019?

### Reference answer (golden set)

A multa não deverá exceder R$ 50.000.000,00 (cinquenta milhões de reais), entre outros critérios possíveis (art. 61, I).

### Cited authoritative text

**`ICVM-607/2019::art-61::inc-i`**

> I – R$ 50.000.000,00 (cinquenta milhões de reais);

### Generated answer

Segundo o art. 61, inciso I, da Instrução CVM nº 607/2019, um dos valores máximos para a penalidade de multa é de R$ 50.000.000,00 (cinquenta milhões de reais) [ICVM-607/2019::art-61::inc-i].

### Your scores

| `q149-raptor` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 5. `q057-graphrag`

- Class: **semantic**
- Strategy: `graphrag`

### Question

Quais informações sobre o tratamento de dados devem ser disponibilizadas ao titular de forma clara e ostensiva, segundo o art. 9º da LGPD?

### Reference answer (golden set)

Devem ser disponibilizadas informações sobre a finalidade específica do tratamento, a forma e duração do tratamento, a identificação do controlador, informações de contato, uso compartilhado de dados, responsabilidades dos agentes, e os direitos do titular (art. 9º, caput).

### Cited authoritative text

_The answer cites nothing._

### Generated answer

O contexto fornecido não contém o artigo 9º da Lei Geral de Proteção de Dados (LGPD), portanto, não é possível responder à sua pergunta com base nas informações disponibilizadas.

### Your scores

| `q057-graphrag` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 6. `q228-parent_child`

- Class: **multi_hop**
- Strategy: `parent_child`

### Question

Como o adiamento de sessão de julgamento por motivo de força maior (art. 50, § 4º) se relaciona com a exigência de presença mínima de membros do Colegiado (art. 50, caput) na ICVM 607/2019?

### Reference answer (golden set)

O art. 50, caput, exige a presença de no mínimo 3 membros do Colegiado para a sessão se realizar; se isso não ocorrer por motivo de força maior, o art. 50, § 4º, permite que a sessão seja transferida para o primeiro dia útil seguinte, sem necessidade de nova convocação.

### Cited authoritative text

**`ICVM-607/2019::art-50`**

> Art. 50.  A sessão será presidida pelo Presidente da CVM ou, na sua ausência ou
> impedimento, por seu substituto eventual e somente será realizada com a presença de no mínimo 3
> (três) membros do Colegiado.
> § 1º  A participação dos membros do Colegiado nas sessões de julgamento poderá ocorrer por
> videoconferência.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 21
> § 2º  No julgamento dos processos administrativos sancionadores em que não comparecer
> nenhum acusado ou nenhum de seus respectivos procuradores constituídos nos autos, a sessão
> poderá realizar-se por meio eletrônico.
> § 3º  Considerando o andamento dos trabalhos da sessão, fica facultado ao Presidente
> suspendê-la e reiniciá-la no dia útil subsequente, independente de nova convocação e publicação.
> § 4º A sessão que não se realizar, por motivo de força maior, poderá ser transferida para o
> primeiro dia útil seguinte, na hora anteriormente marcada, independentemente de nova convocação
> e publicação.

**`ICVM-607/2019::art-50::par-4`**

> § 4º A sessão que não se realizar, por motivo de força maior, poderá ser transferida para o
> primeiro dia útil seguinte, na hora anteriormente marcada, independentemente de nova convocação
> e publicação.

### Generated answer

A Instrução CVM nº 607/2019 estabelece que a sessão de julgamento somente será realizada com a presença de, no mínimo, 3 (três) membros do Colegiado [ICVM-607/2019::art-50].

Caso a sessão não ocorra por motivo de força maior, ela poderá ser transferida para o primeiro dia útil seguinte, no mesmo horário anteriormente marcado, sem a necessidade de nova convocação ou publicação [ICVM-607/2019::art-50::par-4]. Embora o texto não explicite a relação direta entre o quórum e o adiamento, a regra de transferência mantém a sessão agendada, submetendo-a, na nova data, à exigência geral de presença mínima de 3 (três) membros para a sua efetiva realização [ICVM-607/2019::art-50].

### Your scores

| `q228-parent_child` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 7. `q188-graphrag`

- Class: **section_comparative**
- Strategy: `graphrag`

### Question

Compare os efeitos do termo de compromisso (art. 88) com os efeitos do Acordo de Supervisão (art. 92) na ICVM 607/2019.

### Reference answer (golden set)

O termo de compromisso (art. 88) suspende o processo ou impede sua instauração, sem admissão de culpa. Já o Acordo de Supervisão (art. 92) pressupõe confissão da infração pelo investigado, em troca de extinção da ação punitiva ou redução de 1/3 a 2/3 da penalidade, mediante cooperação efetiva na apuração dos fatos.

### Cited authoritative text

**`ICVM-607/2019::art-88::inc-i`**

> I – a suspensão do processo administrativo em curso, pelo prazo estipulado para o
> cumprimento do compromisso; ou

**`ICVM-607/2019::art-88::inc-ii`**

> II – a não instauração de processo administrativo sancionador, nos casos em que a proposta
> for apresentada ainda em fase de apuração ou antes desta.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 35

**`ICVM-607/2019::art-92`**

> Art. 92.  A CVM poderá celebrar acordo administrativo em processo de supervisão (“Acordo
> de Supervisão”) com pessoas naturais ou jurídicas que confessarem a prática de infração às normas
> legais ou regulamentares cujo cumprimento lhe caiba fiscalizar, com extinção de sua ação punitiva
> ou redução de 1/3 (um terço) a 2/3 (dois terços) da penalidade aplicável, mediante efetiva, plena e
> permanente cooperação para a apuração dos fatos, da qual resulte utilidade para o processo, em
> especial:
> I – a identificação dos demais envolvidos na prática da infração, quando couber; e
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 36
> II – a obtenção de informações e de documentos que comprovem a infração noticiada ou sob
> apuração.

**`ICVM-607/2019::art-101::par-3`**

> § 3º  A celebração do Acordo de Supervisão pela CVM suspenderá o prazo prescricional no
> âmbito administrativo com relação ao seu signatário.

### Generated answer

Com base na Instrução CVM nº 607/2019, os efeitos dos dois institutos são distintos:

**Termo de Compromisso (art. 88):**
A celebração do termo de compromisso tem como efeitos:

- A suspensão de processo administrativo em curso, pelo prazo estipulado para o cumprimento do compromisso [ICVM-607/2019::art-88::inc-i]; ou
- A não instauração de processo administrativo sancionador, caso a proposta seja apresentada em fase de apuração ou antes dela [ICVM-607/2019::art-88::inc-ii].

**Acordo de Supervisão (art. 92):**
A celebração do Acordo de Supervisão, que exige a confissão da prática de infração e cooperação efetiva, plena e permanente, resulta em:

- Extinção da ação punitiva da CVM; ou
- Redução de 1/3 (um terço) a 2/3 (dois terços) da penalidade aplicável [ICVM-607/2019::art-92].

Adicionalmente, a celebração do Acordo de Supervisão produz efeitos específicos não mencionados para o termo de compromisso no contexto fornecido, tais como a suspensão do prazo prescricional no âmbito administrativo em relação ao signatário [ICVM-607/2019::art-101::par-3].

### Your scores

| `q188-graphrag` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 8. `q057-sac_contextual`

- Class: **semantic**
- Strategy: `sac_contextual`

### Question

Quais informações sobre o tratamento de dados devem ser disponibilizadas ao titular de forma clara e ostensiva, segundo o art. 9º da LGPD?

### Reference answer (golden set)

Devem ser disponibilizadas informações sobre a finalidade específica do tratamento, a forma e duração do tratamento, a identificação do controlador, informações de contato, uso compartilhado de dados, responsabilidades dos agentes, e os direitos do titular (art. 9º, caput).

### Cited authoritative text

**`LEI-13709/2018::art-9::inc-i`**

> I - finalidade específica do tratamento;

**`LEI-13709/2018::art-9::inc-ii`**

> II - forma e duração do tratamento, observados os segredos comercial e industrial;

**`LEI-13709/2018::art-9::inc-iii`**

> III - identificação do controlador;

**`LEI-13709/2018::art-9::inc-iv`**

> IV - informações de contato do controlador;

**`LEI-13709/2018::art-9::inc-v`**

> V - informações acerca do uso compartilhado de dados pelo controlador e a finalidade;

**`LEI-13709/2018::art-9::inc-vi`**

> VI - responsabilidades dos agentes que realizarão o tratamento; e

**`LEI-13709/2018::art-9::inc-vii`**

> VII - direitos do titular, com menção explícita aos direitos contidos no art. 18 desta Lei.

### Generated answer

Segundo o art. 9º da LGPD, o titular tem direito ao acesso facilitado a informações sobre o tratamento de seus dados, que devem ser disponibilizadas de forma clara, adequada e ostensiva, incluindo:

- A finalidade específica do tratamento [LEI-13709/2018::art-9::inc-i];
- A forma e a duração do tratamento, observados os segredos comercial e industrial [LEI-13709/2018::art-9::inc-ii];
- A identificação do controlador [LEI-13709/2018::art-9::inc-iii];
- As informações de contato do controlador [LEI-13709/2018::art-9::inc-iv];
- Informações acerca do uso compartilhado de dados pelo controlador e a finalidade desse compartilhamento [LEI-13709/2018::art-9::inc-v];
- As responsabilidades dos agentes que realizarão o tratamento [LEI-13709/2018::art-9::inc-vi];
- Os direitos do titular, com menção explícita aos direitos contidos no art. 18 da referida Lei [LEI-13709/2018::art-9::inc-vii].

### Your scores

| `q057-sac_contextual` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 9. `q129-hybrid_rrf`

- Class: **global**
- Strategy: `hybrid_rrf`

### Question

De modo geral, quais são as etapas do rito processual sancionador estabelecido pela ICVM 607/2019 até a apresentação de defesa?

### Reference answer (golden set)

O processo se inicia com a lavratura de termo de acusação (art. 6º), podendo ser precedido de inquérito administrativo (art. 8º); instaura-se com a citação dos acusados (art. 21); o acusado apresenta defesa em até 30 dias (art. 29); e após a defesa, é designado Relator por sorteio (art. 31).

### Cited authoritative text

**`ICVM-607/2019::art-4`**

> Art. 4°  Considerando as informações obtidas na investigação das infrações administrativas,
> as superintendências poderão:
> I – deixar de lavrar termo de acusação nos casos em que:
> a) concluir pela inexistência de irregularidades ou pela extinção da punibilidade; ou
> b) restar demonstrada a pouca relevância da conduta, a baixa expressividade da ameaça ou da
> lesão ao bem jurídico tutelado e a possibilidade de utilização de outros instrumentos ou medidas de
> supervisão que julgarem mais efetivos;
> II – lavrar termo de acusação, nos termos do art. 6º; ou
> III – propor inquérito administrativo destinado a aprofundar a coleta de elementos adicionais à
> verificação da autoria e da materialidade da infração, nos temos do art. 8º.
> § 1º  Na avaliação da relevância da conduta ou da expressividade da ameaça ou lesão ao bem
> jurídico, poderão ser utilizados os seguintes parâmetros, dentre outros:
> I – o grau de reprovabilidade ou da repercussão da conduta;
> II – a expressividade de valores relacionados à conduta;
> III – a expressividade de prejuízos causados a investidores e demais participantes do mercado;
> IV – o impacto da conduta na credibilidade do mercado de capitais;
> V – os antecedentes das pessoas envolvidas;
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 3
> VI – a boa-fé das pessoas envolvidas;
> VII – a regularização da suposta infração pelo administrado; e
> VIII – o ressarcimento dos investidores lesados.
> § 2º  Consideram-se instrumentos e medidas de supervisão, para os fins deste artigo, a
> expedição de ofício de alerta, a atuação prévia e coordenada de instituição autorreguladora, entre
> outros.
> § 3º  A expedição de ofício de alerta à pessoa natural ou jurídica supervisionada, nos termos
> do § 2º, deverá indicar claramente o desvio de conduta verificado e assinalar prazo razoável para a
> devida correção, se aplicável.
> § 4º  Somente caberá recurso da decisão contida no inciso I, do caput, se ausente a
> fundamentação ou caso esteja em desacordo com posicionamento prevalecente no Colegiado.
> § 5º  No recurso de que trata o § 4º, incumbe ao recorrente demonstrar expressamente a
> ausência de fundamentação ou a dissonância em relação ao posicionamento prevalecente do
> Colegiado.
> § 6º  A decisão do Colegiado nas hipóteses de deferimento do recurso previsto no § 4º, não
> determinará a instauração de processo administrativo sancionador, cabendo à superintendência, em
> cada caso, a eventual complementação da fundamentação ou revisão das circunstâncias de fato de
> acordo com o posicionamento prevalecente no Colegiado ou com nova orientação sobre a matéria
> por ele emitida, nos termos do § 8º do art. 4º.
> § 7º  A norma que dispõe sobre o recurso ao Colegiado de decisões proferidas pelos
> superintendentes da CVM aplica-se aos recursos previstos no § 4º exclusivamente no que diz
> respeito aos prazos e procedimentos.
> § 8º  O Colegiado poderá, de ofício ou a pedido da superintendência, conhecer de tema objeto
> de recurso sob a forma de consulta, hipótese na qual deverá manifestar-se sobre a matéria.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 4

**`ICVM-607/2019::art-7`**

> Art. 7º.  Antes da citação dos acusados para apresentação de defesa, a PFE emitirá parecer
> sobre o termo de acusação, no prazo de 30 (trinta) dias contados da data de recebimento do termo
> de acusação, com o seguinte escopo:
> I – exame do cumprimento do art. 5º;
> II – análise objetiva da observância dos requisitos do art. 6º; e
> III – exame da adequação do rito adotado para o processo administrativo sancionador.
> § 1°  Considerando o parecer da PFE, a superintendência tomará as providências que
> considerar cabíveis, podendo, inclusive, arquivar o processo, adequar o rito processual ou realizar
> eventuais ajustes no termo de acusação.
> § 2°  O superintendente deverá justificar a não adoção de eventuais providências
> recomendadas pelo parecer.
> § 3º O parecer da PFE não será obrigatório nos processos administrativos sancionadores
> submetidos ao rito simplificado de que trata o art. 73 desta Instrução.
> § 4º  Sem prejuízo da emissão do parecer de que trata este artigo, as superintendências
> poderão solicitar assessoramento jurídico direto à PFE ainda na fase de instrução.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 6

**`ICVM-607/2019::art-7::par-3`**

> § 3º O parecer da PFE não será obrigatório nos processos administrativos sancionadores
> submetidos ao rito simplificado de que trata o art. 73 desta Instrução.

**`ICVM-607/2019::art-7::par-1`**

> § 1°  Considerando o parecer da PFE, a superintendência tomará as providências que
> considerar cabíveis, podendo, inclusive, arquivar o processo, adequar o rito processual ou realizar
> eventuais ajustes no termo de acusação.

**`ICVM-607/2019::art-21`**

> Art. 21.  Considera-se instaurado o processo administrativo sancionador com a citação dos
> acusados para apresentação de defesa.
> § 1º  A citação conterá:
> I – a identificação do acusado;
> II – a indicação dos fatos imputados ao acusado;
> III – a finalidade da citação;
> IV – o prazo para a apresentação de defesa;
> V – a informação da continuidade do processo, independentemente de seu comparecimento;
> VI – o dever do acusado, ou de procurador por ele constituído, de se cadastrar no sistema de
> processo eletrônico existente na página da CVM na rede mundial de computadores para fins de
> acesso aos autos e posterior acompanhamento do andamento do processo; e
> VII – o aviso de que o acusado poderá propor a celebração de termo de compromisso, em
> conformidade com o disposto no Capítulo IV desta Instrução.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 10
> § 2º  O requisito de que trata o inciso II do § 1º poderá ser atendido por meio da juntada do
> termo ou da peça de acusação.
> § 3º  Quando se tratar de processo em meio físico, a citação deverá conter, além dos requisitos
> previstos no § 1º, o termo de acusação ou a peça acusatória, a depender do caso.
> § 4º  O acusado que, embora citado, não apresentar defesa será considerado revel.

**`ICVM-607/2019::art-21::par-1`**

> § 1º  A citação conterá:
> I – a identificação do acusado;
> II – a indicação dos fatos imputados ao acusado;
> III – a finalidade da citação;
> IV – o prazo para a apresentação de defesa;
> V – a informação da continuidade do processo, independentemente de seu comparecimento;
> VI – o dever do acusado, ou de procurador por ele constituído, de se cadastrar no sistema de
> processo eletrônico existente na página da CVM na rede mundial de computadores para fins de
> acesso aos autos e posterior acompanhamento do andamento do processo; e
> VII – o aviso de que o acusado poderá propor a celebração de termo de compromisso, em
> conformidade com o disposto no Capítulo IV desta Instrução.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 10

**`ICVM-607/2019::art-29`**

> Art. 29.  O acusado deverá apresentar sua defesa por escrito no prazo de 30 (trinta) dias após a
> citação, oportunidade em que deverá juntar os documentos destinados a provar suas alegações e
> especificar as demais provas que pretenda produzir, observado o disposto nos arts. 42 e 43 desta
> Instrução.
> § 1º  O interessado deverá manifestar sua intenção de celebrar termo de compromisso até o
> término do prazo para a apresentação de defesa, e sem prejuízo do ônus de apresentação desta.
> § 2º  A manifestação de intenção ou a apresentação de proposta de termo de compromisso não
> suspende nem interrompe o prazo para apresentação da defesa.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 14
> § 3º  Serão computados individualmente os prazos para todas as manifestações dos acusados,
> sendo admitida uma única prorrogação pelo mesmo prazo, diante de pedido devidamente
> fundamentado.
> § 4º  Os acusados que constituírem o mesmo procurador e apresentarem defesa conjunta terão
> o mesmo prazo para se manifestarem nos autos, contado da citação que for efetivada por último, nos
> termos do art. 29, § 3º, desta Instrução.
> § 5º  Nos processos sancionadores instaurados em desfavor de múltiplos acusados, as defesas
> serão consideradas sigilosas e não serão fornecidas a terceiros ou a outros acusados até o
> encerramento do último prazo de apresentação de defesa.

### Generated answer

Com base na Instrução CVM nº 607/2019, as etapas do rito processual sancionador até a apresentação da defesa são as seguintes:

1. **Investigação e Decisão da Superintendência:** Após a investigação das infrações, a superintendência pode optar por lavrar o termo de acusação, propor inquérito administrativo ou deixar de lavrar o termo (em casos de inexistência de irregularidades ou pouca relevância da conduta) [ICVM-607/2019::art-4].
2. **Parecer da PFE:** Antes da citação dos acusados, a Procuradoria Federal Especializada (PFE) deve emitir um parecer sobre o termo de acusação no prazo de 30 dias, examinando o cumprimento dos requisitos legais e a adequação do rito [ICVM-607/2019::art-7]. Este parecer não é obrigatório em processos sob rito simplificado [ICVM-607/2019::art-7::par-3].
3. **Providências da Superintendência:** Com base no parecer da PFE, a superintendência pode arquivar o processo, ajustar o termo de acusação ou adequar o rito processual [ICVM-607/2019::art-7::par-1].
4. **Citação:** O processo administrativo sancionador é considerado instaurado com a citação dos acusados [ICVM-607/2019::art-21]. A citação deve conter informações como a identificação do acusado, os fatos imputados, o prazo para defesa e o aviso sobre a possibilidade de propor termo de compromisso [ICVM-607/2019::art-21::par-1].
5. **Apresentação de Defesa:** O acusado deve apresentar sua defesa por escrito no prazo de 30 dias após a citação, juntando os documentos necessários e especificando as provas que pretende produzir [ICVM-607/2019::art-29].

### Your scores

| `q129-hybrid_rrf` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 10. `q017-hybrid_rrf`

- Class: **unanswerable**  ·  **UNANSWERABLE**
- Strategy: `hybrid_rrf`

### Question

Qual é o valor da multa pecuniária prevista na RES-CMN-5274/2025 para o descumprimento do prazo de adaptação até março de 2026?

### Reference answer (golden set)

A RES-CMN-5274/2025 não estabelece um valor de multa pecuniária para o descumprimento desse prazo; o texto da resolução não trata dessa questão.

### Cited authoritative text

**`RES-CMN-5274/2025::art-2`**

> Art. 2º As instituições em funcionamento na data da entrada em vigor desta Resolução devem promover as adaptações necessárias à adequação ao disposto nesta Resolução até 1º de março de 2026.

### Generated answer

O contexto fornecido não contém informações sobre o valor de multas pecuniárias específicas para o descumprimento do prazo de adaptação estabelecido pela Resolução CMN nº 5274/2025. O documento apenas determina que as instituições em funcionamento devem promover as adaptações necessárias até 1º de março de 2026 [RES-CMN-5274/2025::art-2].

### Your scores

| `q017-hybrid_rrf` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |
| abstention | | |

---

## 11. `q228-dense`

- Class: **multi_hop**
- Strategy: `dense`

### Question

Como o adiamento de sessão de julgamento por motivo de força maior (art. 50, § 4º) se relaciona com a exigência de presença mínima de membros do Colegiado (art. 50, caput) na ICVM 607/2019?

### Reference answer (golden set)

O art. 50, caput, exige a presença de no mínimo 3 membros do Colegiado para a sessão se realizar; se isso não ocorrer por motivo de força maior, o art. 50, § 4º, permite que a sessão seja transferida para o primeiro dia útil seguinte, sem necessidade de nova convocação.

### Cited authoritative text

**`ICVM-607/2019::art-50`**

> Art. 50.  A sessão será presidida pelo Presidente da CVM ou, na sua ausência ou
> impedimento, por seu substituto eventual e somente será realizada com a presença de no mínimo 3
> (três) membros do Colegiado.
> § 1º  A participação dos membros do Colegiado nas sessões de julgamento poderá ocorrer por
> videoconferência.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 21
> § 2º  No julgamento dos processos administrativos sancionadores em que não comparecer
> nenhum acusado ou nenhum de seus respectivos procuradores constituídos nos autos, a sessão
> poderá realizar-se por meio eletrônico.
> § 3º  Considerando o andamento dos trabalhos da sessão, fica facultado ao Presidente
> suspendê-la e reiniciá-la no dia útil subsequente, independente de nova convocação e publicação.
> § 4º A sessão que não se realizar, por motivo de força maior, poderá ser transferida para o
> primeiro dia útil seguinte, na hora anteriormente marcada, independentemente de nova convocação
> e publicação.

**`ICVM-607/2019::art-50::par-4`**

> § 4º A sessão que não se realizar, por motivo de força maior, poderá ser transferida para o
> primeiro dia útil seguinte, na hora anteriormente marcada, independentemente de nova convocação
> e publicação.

### Generated answer

De acordo com a Instrução CVM nº 607/2019, a realização de uma sessão de julgamento exige a presença de, no mínimo, 3 (três) membros do Colegiado [ICVM-607/2019::art-50].

Caso a sessão não ocorra por motivo de força maior, ela poderá ser transferida para o primeiro dia útil seguinte, no mesmo horário anteriormente marcado, sem a necessidade de nova convocação ou publicação [ICVM-607/2019::art-50::par-4]. Embora o texto não estabeleça uma relação direta de dependência entre o motivo de força maior e a falta de quórum, a exigência de presença mínima de 3 membros permanece como regra geral para a realização da sessão [ICVM-607/2019::art-50].

### Your scores

| `q228-dense` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 12. `q188-sac`

- Class: **section_comparative**
- Strategy: `sac`

### Question

Compare os efeitos do termo de compromisso (art. 88) com os efeitos do Acordo de Supervisão (art. 92) na ICVM 607/2019.

### Reference answer (golden set)

O termo de compromisso (art. 88) suspende o processo ou impede sua instauração, sem admissão de culpa. Já o Acordo de Supervisão (art. 92) pressupõe confissão da infração pelo investigado, em troca de extinção da ação punitiva ou redução de 1/3 a 2/3 da penalidade, mediante cooperação efetiva na apuração dos fatos.

### Cited authoritative text

**`ICVM-607/2019::art-88::inc-i`**

> I – a suspensão do processo administrativo em curso, pelo prazo estipulado para o
> cumprimento do compromisso; ou

**`ICVM-607/2019::art-88::inc-ii`**

> II – a não instauração de processo administrativo sancionador, nos casos em que a proposta
> for apresentada ainda em fase de apuração ou antes desta.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 35

**`ICVM-607/2019::art-92`**

> Art. 92.  A CVM poderá celebrar acordo administrativo em processo de supervisão (“Acordo
> de Supervisão”) com pessoas naturais ou jurídicas que confessarem a prática de infração às normas
> legais ou regulamentares cujo cumprimento lhe caiba fiscalizar, com extinção de sua ação punitiva
> ou redução de 1/3 (um terço) a 2/3 (dois terços) da penalidade aplicável, mediante efetiva, plena e
> permanente cooperação para a apuração dos fatos, da qual resulte utilidade para o processo, em
> especial:
> I – a identificação dos demais envolvidos na prática da infração, quando couber; e
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 36
> II – a obtenção de informações e de documentos que comprovem a infração noticiada ou sob
> apuração.

**`ICVM-607/2019::art-93::par-2`**

> § 2º  A proposta de Acordo de Supervisão não obstará as atividades de investigação e
> instrução nem a tramitação do processo administrativo sancionador que porventura já tenha sido
> anteriormente instaurado para a apuração das condutas narradas na proposta.

### Generated answer

Os efeitos dos dois institutos previstos na Instrução CVM nº 607/2019 são distintos conforme a natureza do acordo:

**Termo de Compromisso (art. 88):**
A celebração do termo de compromisso produz os seguintes efeitos:

- A suspensão de um processo administrativo que já esteja em curso, durante o prazo estipulado para o cumprimento do compromisso [ICVM-607/2019::art-88::inc-i]; ou
- A não instauração de processo administrativo sancionador, caso a proposta seja apresentada antes ou durante a fase de apuração [ICVM-607/2019::art-88::inc-ii].

**Acordo de Supervisão (art. 92):**
A celebração do Acordo de Supervisão, que exige a confissão da prática de infração e cooperação efetiva, plena e permanente, produz os seguintes efeitos:

- A extinção da ação punitiva da CVM; ou
- A redução de 1/3 (um terço) a 2/3 (dois terços) da penalidade aplicável [ICVM-607/2019::art-92].

Vale ressaltar que, diferentemente do termo de compromisso, a proposta de Acordo de Supervisão não impede as atividades de investigação, instrução ou a tramitação de processo administrativo sancionador que já tenha sido instaurado para apurar as condutas narradas [ICVM-607/2019::art-93::par-2].

### Your scores

| `q188-sac` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 13. `q162-sac`

- Class: **global**
- Strategy: `sac`

### Question

De modo geral, como a ICVM 607/2019 estrutura o cálculo e a dosimetria das penalidades aplicáveis pela CVM?

### Reference answer (golden set)

A CVM pode impor sete tipos de penalidade (art. 60), com limites para a multa (art. 61); o Colegiado fixa uma pena-base observando proporcionalidade e capacidade econômica (art. 63), sobre a qual incidem circunstâncias agravantes, que aumentam a pena (art. 65), e atenuantes, que a reduzem (art. 66).

### Cited authoritative text

**`ICVM-607/2019::art-63`**

> Art. 63.  Na fixação da pena-base, o Colegiado observará os princípios da proporcionalidade e
> da razoabilidade, bem como a capacidade econômica do infrator e os motivos que justifiquem a
> imposição da penalidade.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 25
> § 1º  Se adotado o critério de que trata o art. 61, inciso I, a pena-base da multa deverá
> observar os limites aplicáveis a cada infração, previstos no Anexo 63, sem prejuízo da aplicação
> cumulativa de outras modalidades de pena descritas no art. 60 desta Instrução.
> § 2º  Na hipótese do § 1º em que a infração não esteja prevista no Anexo 63, o Colegiado
> deverá, com base na gravidade da conduta, enquadrá-la em um dos grupos previstos no referido
> Anexo.
> § 3º  A pena-base das penalidades descritas no art. 60, incisos III a VI, deverá ser fixada em
> meses e não poderá ser superior a 10 (dez) anos.
> § 4º  A pena-base da penalidade descrita no art. 60, inciso VII, deverá ser fixada em meses e
> não poderá ser superior a 5 (cinco) anos.

**`ICVM-607/2019::art-63::par-1`**

> § 1º  Se adotado o critério de que trata o art. 61, inciso I, a pena-base da multa deverá
> observar os limites aplicáveis a cada infração, previstos no Anexo 63, sem prejuízo da aplicação
> cumulativa de outras modalidades de pena descritas no art. 60 desta Instrução.

**`ICVM-607/2019::art-63::par-2`**

> § 2º  Na hipótese do § 1º em que a infração não esteja prevista no Anexo 63, o Colegiado
> deverá, com base na gravidade da conduta, enquadrá-la em um dos grupos previstos no referido
> Anexo.

**`ICVM-607/2019::art-63::par-3`**

> § 3º  A pena-base das penalidades descritas no art. 60, incisos III a VI, deverá ser fixada em
> meses e não poderá ser superior a 10 (dez) anos.

**`ICVM-607/2019::art-63::par-4`**

> § 4º  A pena-base da penalidade descrita no art. 60, inciso VII, deverá ser fixada em meses e
> não poderá ser superior a 5 (cinco) anos.

**`ICVM-607/2019::art-62`**

> Art. 62.  Na dosimetria da pena, salvo se aplicada a penalidade de advertência, o Colegiado
> fixará inicialmente a pena-base, aplicando na sequência as circunstâncias agravantes e atenuantes,
> bem como a causa de redução da pena, nessa ordem.
> Parágrafo único.  O Colegiado considerará na dosimetria as demais sanções relativas aos
> mesmos fatos, aplicadas definitivamente por outras autoridades, cabendo ao acusado demonstrar,
> até o julgamento do processo pelo Colegiado, o cabimento dessa circunstância.

**`ICVM-607/2019::art-65::par-1`**

> § 1º  A penalidade de multa será acrescida em até 25% (vinte e cinco por cento) para cada
> agravante verificada.

**`ICVM-607/2019::art-65::par-2`**

> § 2º  As penalidades de suspensão, inabilitação e proibição serão acrescidas em até 25% (vinte
> e cinco por cento) para cada agravante verificada, considerando-se o número de meses da pena-base
> e desprezando-se as frações.

**`ICVM-607/2019::art-62::par-unico`**

> Parágrafo único.  O Colegiado considerará na dosimetria as demais sanções relativas aos
> mesmos fatos, aplicadas definitivamente por outras autoridades, cabendo ao acusado demonstrar,
> até o julgamento do processo pelo Colegiado, o cabimento dessa circunstância.

**`ICVM-607/2019::art-4::par-1`**

> § 1º  Na avaliação da relevância da conduta ou da expressividade da ameaça ou lesão ao bem
> jurídico, poderão ser utilizados os seguintes parâmetros, dentre outros:
> I – o grau de reprovabilidade ou da repercussão da conduta;
> II – a expressividade de valores relacionados à conduta;
> III – a expressividade de prejuízos causados a investidores e demais participantes do mercado;
> IV – o impacto da conduta na credibilidade do mercado de capitais;
> V – os antecedentes das pessoas envolvidas;
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 3
> VI – a boa-fé das pessoas envolvidas;
> VII – a regularização da suposta infração pelo administrado; e
> VIII – o ressarcimento dos investidores lesados.

### Generated answer

A Instrução CVM 607/2019 estrutura a dosimetria das penalidades seguindo uma ordem específica de aplicação, conforme detalhado abaixo:

**1. Fixação da Pena-Base**

- **Critérios:** Ao fixar a pena-base, o Colegiado deve observar os princípios da proporcionalidade e da razoabilidade, a capacidade econômica do infrator e os motivos que justificam a penalidade [ICVM-607/2019::art-63].
- **Limites:** Para a multa, deve-se observar os limites previstos no Anexo 63 da Instrução [ICVM-607/2019::art-63::par-1]. Caso a infração não esteja no Anexo, o Colegiado deve enquadrá-la em um dos grupos previstos com base na gravidade da conduta [ICVM-607/2019::art-63::par-2].
- **Prazos:** Para as penalidades de inabilitação, suspensão e proibição (incisos III a VI do art. 60), a pena-base não pode exceder 10 anos [ICVM-607/2019::art-63::par-3]. Para a proibição de atuar no mercado (inciso VII do art. 60), o limite é de 5 anos [ICVM-607/2019::art-63::par-4].

**2. Ordem de Aplicação**
Salvo no caso de advertência, a dosimetria segue a seguinte sequência:

1. Fixação da pena-base [ICVM-607/2019::art-62].
2. Aplicação das circunstâncias agravantes e atenuantes [ICVM-607/2019::art-62].
3. Aplicação da causa de redução da pena [ICVM-607/2019::art-62].

**3. Ajustes por Agravantes**

- **Multa:** Pode ser acrescida em até 25% para cada agravante verificada [ICVM-607/2019::art-65::par-1].
- **Suspensão, Inabilitação e Proibição:** Também podem ser acrescidas em até 25% por agravante, calculadas sobre o número de meses da pena-base e desprezando-se as frações [ICVM-607/2019::art-65::par-2].

**4. Considerações Adicionais**

- **Outras sanções:** O Colegiado deve considerar na dosimetria sanções aplicadas definitivamente por outras autoridades sobre os mesmos fatos, cabendo ao acusado demonstrar essa circunstância até o julgamento [ICVM-607/2019::art-62::par-unico].
- **Relevância:** Na avaliação da conduta, podem ser considerados parâmetros como o grau de reprovabilidade, valores envolvidos, prejuízos a investidores, impacto na credibilidade do mercado, antecedentes, boa-fé, regularização da infração e ressarcimento aos lesados [ICVM-607/2019::art-4::par-1].

### Your scores

| `q162-sac` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 14. `q091-raptor`

- Class: **unanswerable**  ·  **UNANSWERABLE**
- Strategy: `raptor`

### Question

Qual é o valor mínimo da multa diária prevista no art. 52, inciso III, da LGPD?

### Reference answer (golden set)

O art. 52, III, não estabelece um valor mínimo para a multa diária; apenas determina que ela observe o limite total do inciso II (até R$ 50 milhões por infração), sem fixar piso mínimo.

### Cited authoritative text

**`LEI-13709/2018::art-54`**

> Art. 54. O valor da sanção de multa diária aplicável às infrações a esta Lei deve observar a gravidade da falta e a extensão do dano ou prejuízo causado e ser fundamentado pela autoridade nacional.
> Parágrafo único. A intimação da sanção de multa diária deverá conter, no mínimo, a descrição da obrigação imposta, o prazo razoável e estipulado pelo órgão para o seu cumprimento e o valor da multa diária a ser aplicada pelo seu descumprimento.   (Vigência)
> DA AUTORIDADE NACIONAL DE PROTEÇÃO DE DADOS (ANPD) E DO CONSELHO NACIONAL DE PROTEÇÃO DE DADOS PESSOAIS E DA PRIVACIDADE
> Da Autoridade Nacional de Proteção de Dados (ANPD)
> (Redação dada pela Medida Provisória nº 1.317, de 2025)
> DA AGÊNCIA NACIONAL DE PROTEÇÃO DE DADOS E DO CONSELHO NACIONAL DE PROTEÇÃO DE DADOS PESSOAIS E DA PRIVACIDADE
> Da Agência Nacional de Proteção de Dados
> (Redação dada pela Lei nº 15.352, de 2026)
> DA AGÊNCIA NACIONAL DE PROTEÇÃO DE DADOS E DO CONSELHO NACIONAL DE PROTEÇÃO DE DADOS PESSOAIS E DA PRIVACIDADE’
> Da Agência Nacional de Proteção de  Dados

### Generated answer

O contexto fornecido não estabelece um valor mínimo para a multa diária prevista no art. 52, inciso III, da LGPD. O texto menciona apenas que a multa diária deve observar o limite total estabelecido para a multa simples (limitada a R$ 50.000.000,00 por infração) [LEI-13709/2018::art-52::inc-ii, LEI-13709/2018::art-52::inc-iii] e que o seu valor deve ser fundamentado pela autoridade nacional, considerando a gravidade da falta e a extensão do dano ou prejuízo causado [LEI-13709/2018::art-54].

### Your scores

| `q091-raptor` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |
| abstention | | |

---

## 15. `q147-reranked`

- Class: **exact_factual**
- Strategy: `reranked`

### Question

Quais penalidades a CVM pode impor aos infratores, segundo o art. 60 da ICVM 607/2019?

### Reference answer (golden set)

Advertência, multa, inabilitação temporária para cargos, suspensão de autorização ou registro, inabilitação para atividades da Lei 6.385/1976, proibição temporária de praticar atividades, e proibição temporária de atuar no mercado de valores mobiliários (art. 60).

### Cited authoritative text

**`ICVM-607/2019::art-60::inc-i`**

> I – advertência;

**`ICVM-607/2019::art-60::inc-ii`**

> II – multa;

**`ICVM-607/2019::art-60::inc-iii`**

> III – inabilitação temporária, até o máximo de 20 (vinte) anos, para o exercício de cargo de
> administrador ou de conselheiro fiscal de companhia aberta, de entidade do sistema de distribuição
> ou de outras entidades que dependam de autorização ou registro na Comissão de Valores
> Mobiliários;

**`ICVM-607/2019::art-60::inc-iv`**

> IV – suspensão da autorização ou registro para o exercício das atividades de que trata a Lei nº
> 6.385, de 1976;

**`ICVM-607/2019::art-60::inc-v`**

> V – inabilitação temporária, até o máximo de 20 (vinte) anos, para o exercício das atividades
> de que trata a Lei nº 6.385, de 7 de dezembro de 1976;

**`ICVM-607/2019::art-60::inc-vi`**

> VI – proibição temporária, até o máximo de 20 (vinte) anos, de praticar determinadas
> atividades ou operações, para os integrantes do sistema de distribuição ou de outras entidades que
> dependam de autorização ou registro na Comissão de Valores Mobiliários; e
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 24

**`ICVM-607/2019::art-60::inc-vii`**

> VII – proibição temporária, até o máximo de 10 (dez) anos, de atuar, direta ou indiretamente,
> em uma ou mais modalidades de operação no mercado de valores mobiliários.

### Generated answer

Segundo o art. 60 da Instrução CVM nº 607/2019, a CVM pode impor as seguintes penalidades, de forma isolada ou cumulativa:

- **Advertência** [ICVM-607/2019::art-60::inc-i];
- **Multa** [ICVM-607/2019::art-60::inc-ii];
- **Inabilitação temporária**, até o máximo de 20 anos, para o exercício de cargo de administrador ou de conselheiro fiscal de companhia aberta, de entidade do sistema de distribuição ou de outras entidades que dependam de autorização ou registro na CVM [ICVM-607/2019::art-60::inc-iii];
- **Suspensão da autorização ou registro** para o exercício das atividades previstas na Lei nº 6.385/1976 [ICVM-607/2019::art-60::inc-iv];
- **Inabilitação temporária**, até o máximo de 20 anos, para o exercício das atividades previstas na Lei nº 6.385/1976 [ICVM-607/2019::art-60::inc-v];
- **Proibição temporária**, até o máximo de 20 anos, de praticar determinadas atividades ou operações, para integrantes do sistema de distribuição ou outras entidades que dependam de autorização ou registro na CVM [ICVM-607/2019::art-60::inc-vi];
- **Proibição temporária**, até o máximo de 10 anos, de atuar, direta ou indiretamente, em uma ou mais modalidades de operação no mercado de valores mobiliários [ICVM-607/2019::art-60::inc-vii].

### Your scores

| `q147-reranked` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 16. `q029-sac`

- Class: **exact_factual**
- Strategy: `sac`

### Question

Quais resoluções são revogadas pela RES-CMN-4893/2021, segundo seu art. 27?

### Reference answer (golden set)

São revogadas a Resolução nº 4.658, de 26 de abril de 2018, e a Resolução nº 4.752, de 26 de setembro de 2019 (art. 27).

### Cited authoritative text

**`RES-CMN-4893/2021::art-27::inc-i`**

> I - a Resolução nº 4.658, de 26 de abril de 2018; e

**`RES-CMN-4893/2021::art-27::inc-ii`**

> II - a Resolução nº 4.752, de 26 de setembro de 2019.

### Generated answer

De acordo com o art. 27 da Resolução CMN nº 4.893/2021, as resoluções revogadas são:

- Resolução nº 4.658, de 26 de abril de 2018 [RES-CMN-4893/2021::art-27::inc-i];
- Resolução nº 4.752, de 26 de setembro de 2019 [RES-CMN-4893/2021::art-27::inc-ii].

### Your scores

| `q029-sac` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 17. `q086-contextual`

- Class: **numeric_tabular**
- Strategy: `contextual`

### Question

Qual é o valor máximo da multa simples prevista no art. 52, inciso II, da LGPD, por infração?

### Reference answer (golden set)

Multa simples de até 2% do faturamento da pessoa jurídica de direito privado, grupo ou conglomerado no Brasil no último exercício, excluídos os tributos, limitada, no total, a R$ 50.000.000,00 (cinquenta milhões de reais) por infração (art. 52, II).

### Cited authoritative text

**`LEI-13709/2018::art-52::inc-ii`**

> II - multa simples, de até 2% (dois por cento) do faturamento da pessoa jurídica de direito privado, grupo ou conglomerado no Brasil no seu último exercício, excluídos os tributos, limitada, no total, a R$ 50.000.000,00 (cinquenta milhões de reais) por infração;

### Generated answer

O valor máximo da multa simples prevista no art. 52, inciso II, da LGPD é de R$ 50.000.000,00 (cinquenta milhões de reais) por infração [LEI-13709/2018::art-52::inc-ii].

### Your scores

| `q086-contextual` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 18. `q152-dense`

- Class: **numeric_tabular**
- Strategy: `dense`

### Question

Qual o limite máximo, em anos, para a pena-base das penalidades descritas no art. 60, incisos III a VI, segundo o art. 63, § 3º da ICVM 607/2019?

### Reference answer (golden set)

Não poderá ser superior a 10 (dez) anos (art. 63, § 3º).

### Cited authoritative text

**`ICVM-607/2019::art-63::par-3`**

> § 3º  A pena-base das penalidades descritas no art. 60, incisos III a VI, deverá ser fixada em
> meses e não poderá ser superior a 10 (dez) anos.

### Generated answer

O limite máximo para a pena-base das penalidades descritas nos incisos III a VI do art. 60 da Instrução CVM nº 607/2019 é de 10 (dez) anos [ICVM-607/2019::art-63::par-3].

### Your scores

| `q152-dense` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 19. `q053-contextual`

- Class: **semantic**
- Strategy: `contextual`

### Question

O que caracteriza o princípio da necessidade no tratamento de dados pessoais, segundo o art. 6º, inciso III da LGPD?

### Reference answer (golden set)

É a limitação do tratamento ao mínimo necessário para a realização de suas finalidades, com abrangência dos dados pertinentes, proporcionais e não excessivos em relação às finalidades do tratamento (art. 6º, III).

### Cited authoritative text

**`LEI-13709/2018::art-6::inc-iii`**

> III - necessidade: limitação do tratamento ao mínimo necessário para a realização de suas finalidades, com abrangência dos dados pertinentes, proporcionais e não excessivos em relação às finalidades do tratamento de dados;

### Generated answer

Segundo o art. 6º, inciso III da LGPD, o princípio da necessidade caracteriza-se pela limitação do tratamento de dados ao mínimo necessário para a realização de suas finalidades, abrangendo dados que sejam pertinentes, proporcionais e não excessivos em relação a tais finalidades [LEI-13709/2018::art-6::inc-iii].

### Your scores

| `q053-contextual` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 20. `q160-hybrid_rrf`

- Class: **multi_hop**
- Strategy: `hybrid_rrf`

### Question

Como as circunstâncias agravantes do art. 65 se relacionam com o cálculo da pena-base previsto no art. 63 da ICVM 607/2019?

### Reference answer (golden set)

O art. 63 determina que o Colegiado fixe a pena-base observando proporcionalidade, razoabilidade e capacidade econômica do infrator; sobre essa pena-base incidem as agravantes do art. 65, cada uma acrescendo até 25% ao valor da multa (art. 65, § 1º).

### Cited authoritative text

**`ICVM-607/2019::art-62, ICVM-607/2019::art-63`**

> _(not resolvable - the cited ID does not exist in the parsed corpus)_

**`ICVM-607/2019::art-65::par-1`**

> § 1º  A penalidade de multa será acrescida em até 25% (vinte e cinco por cento) para cada
> agravante verificada.

**`ICVM-607/2019::art-65::par-2`**

> § 2º  As penalidades de suspensão, inabilitação e proibição serão acrescidas em até 25% (vinte
> e cinco por cento) para cada agravante verificada, considerando-se o número de meses da pena-base
> e desprezando-se as frações.

**`ICVM-607/2019::art-65`**

> Art. 65.  São circunstâncias agravantes, quando não constituem ou qualificam a infração:
> I – a prática sistemática ou reiterada da conduta irregular;
> II – o elevado prejuízo causado;
> III – a expressiva vantagem auferida ou pretendida pelo infrator;
> IV – a existência de dano relevante à imagem do mercado de valores mobiliários ou do
> segmento em que atua;
> V – o cometimento de infração mediante ardil, fraude ou simulação;
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 26
> VI – o comprometimento ou risco de comprometimento da solvência do emissor;
> VII – a violação de deveres fiduciários decorrentes do cargo, posição ou função que ocupa; e
> VIII – a ocultação de provas da infração mediante ardil, fraude ou simulação.
> § 1º  A penalidade de multa será acrescida em até 25% (vinte e cinco por cento) para cada
> agravante verificada.
> § 2º  As penalidades de suspensão, inabilitação e proibição serão acrescidas em até 25% (vinte
> e cinco por cento) para cada agravante verificada, considerando-se o número de meses da pena-base
> e desprezando-se as frações.
> § 3º  Ocorrerá reincidência quando o agente comete nova infração depois de ter sido punido
> por força de decisão administrativa definitiva, salvo se decorridos 5 (cinco) anos do cumprimento
> da respectiva punição ou da extinção da pena.

### Generated answer

De acordo com a Instrução CVM 607/2019, a dosimetria da pena segue uma ordem específica: o Colegiado fixa inicialmente a pena-base (observando os princípios da proporcionalidade, razoabilidade, capacidade econômica do infrator e os motivos da penalidade) e, na sequência, aplica as circunstâncias agravantes e atenuantes [ICVM-607/2019::art-62, ICVM-607/2019::art-63].

Em relação ao cálculo, as circunstâncias agravantes do art. 65 impactam a pena da seguinte forma:

- **Multa:** A penalidade de multa é acrescida em até 25% para cada agravante verificada [ICVM-607/2019::art-65::par-1].
- **Suspensão, inabilitação e proibição:** Estas penalidades são acrescidas em até 25% para cada agravante verificada, calculando-se o acréscimo sobre o número de meses da pena-base e desprezando-se as frações [ICVM-607/2019::art-65::par-2].

Vale ressaltar que as agravantes só são aplicadas quando não constituem ou qualificam a própria infração [ICVM-607/2019::art-65].

### Your scores

| `q160-hybrid_rrf` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 21. `q011-reranked`

- Class: **section_comparative**
- Strategy: `reranked`

### Question

Compare as regras de fornecimento de informações sigilosas ao Poder Judiciário (art. 3º) e ao Poder Legislativo Federal (art. 4º) na LC 105/2001.

### Reference answer (golden set)

Ao Poder Judiciário, as informações ordenadas são prestadas preservando o caráter sigiloso mediante acesso restrito às partes do processo (art. 3º). Ao Poder Legislativo Federal, as informações e documentos sigilosos são fornecidos quando fundamentadamente necessários ao exercício de competências constitucionais e legais, sendo que, no caso de comissões parlamentares de inquérito, as solicitações devem ser previamente aprovadas pelo plenário da respectiva Casa ou comissão (art. 4º).

### Cited authoritative text

**`LC-105/2001::art-3`**

> Art. 3o Serão prestadas pelo Banco Central do Brasil, pela Comissão de Valores Mobiliários e pelas instituições financeiras as informações ordenadas pelo Poder Judiciário, preservado o seu caráter sigiloso mediante acesso restrito às partes, que delas não poderão servir-se para fins estranhos à lide.
> § 1o Dependem de prévia autorização do Poder Judiciário a prestação de informações e o fornecimento de documentos sigilosos solicitados por comissão de inquérito administrativo destinada a apurar responsabilidade de servidor público por infração praticada no exercício de suas atribuições, ou que tenha relação com as atribuições do cargo em que se encontre investido.
> § 2o Nas hipóteses do § 1o, o requerimento de quebra de sigilo independe da existência de processo judicial em curso.
> § 3o Além dos casos previstos neste artigo o Banco Central do Brasil e a Comissão de Valores Mobiliários fornecerão à Advocacia-Geral da União as informações e os documentos necessários à defesa da União nas ações em que seja parte.

**`LC-105/2001::art-4`**

> Art. 4o O Banco Central do Brasil e a Comissão de Valores Mobiliários, nas áreas de suas atribuições, e as instituições financeiras fornecerão ao Poder Legislativo Federal as informações e os documentos sigilosos que, fundamentadamente, se fizerem necessários ao exercício de suas respectivas competências constitucionais e legais.
> § 1o As comissões parlamentares de inquérito, no exercício de sua competência constitucional e legal de ampla investigação, obterão as informações e documentos sigilosos de que necessitarem, diretamente das instituições financeiras, ou por intermédio do Banco Central do Brasil ou da Comissão de Valores Mobiliários.
> § 2o As solicitações de que trata este artigo deverão ser previamente aprovadas pelo Plenário da Câmara dos Deputados, do Senado Federal, ou do plenário de suas respectivas comissões parlamentares de inquérito.

**`LC-105/2001::art-4::par-1`**

> § 1o As comissões parlamentares de inquérito, no exercício de sua competência constitucional e legal de ampla investigação, obterão as informações e documentos sigilosos de que necessitarem, diretamente das instituições financeiras, ou por intermédio do Banco Central do Brasil ou da Comissão de Valores Mobiliários.

**`LC-105/2001::art-4::par-2`**

> § 2o As solicitações de que trata este artigo deverão ser previamente aprovadas pelo Plenário da Câmara dos Deputados, do Senado Federal, ou do plenário de suas respectivas comissões parlamentares de inquérito.

### Generated answer

As regras para o fornecimento de informações sigilosas ao Poder Judiciário e ao Poder Legislativo Federal, conforme a LC 105/2001, apresentam as seguintes características:

**Fornecimento ao Poder Judiciário (Art. 3º):**

- **Abrangência:** As informações são prestadas pelo Banco Central do Brasil, pela Comissão de Valores Mobiliários (CVM) e pelas instituições financeiras mediante ordem do Poder Judiciário [LC-105/2001::art-3].
- **Sigilo:** O caráter sigiloso das informações é preservado, com acesso restrito às partes, que não podem utilizá-las para fins estranhos à lide [LC-105/2001::art-3].
- **Comissões de Inquérito Administrativo:** O fornecimento de documentos sigilosos para comissões de inquérito administrativo (que apurem responsabilidade de servidor público) depende de prévia autorização judicial, sendo que, nestes casos, o requerimento independe da existência de processo judicial em curso [LC-105/2001::art-3::par-1, LC-105/2001::art-3::par-2].

**Fornecimento ao Poder Legislativo Federal (Art. 4º):**

- **Abrangência:** O Banco Central, a CVM e as instituições financeiras fornecem as informações e documentos sigilosos que se fizerem necessários ao exercício das competências constitucionais e legais do Poder Legislativo [LC-105/2001::art-4].
- **Fundamentação:** As solicitações devem ser fundamentadas [LC-105/2001::art-4].
- **Comissões Parlamentares de Inquérito (CPIs):** As CPIs podem obter as informações diretamente das instituições financeiras ou por intermédio do Banco Central ou da CVM [LC-105/2001::art-4::par-1].
- **Aprovação Prévia:** As solicitações feitas ao Poder Legislativo devem ser previamente aprovadas pelo Plenário da Câmara dos Deputados, do Senado Federal ou pelo plenário de suas respectivas CPIs [LC-105/2001::art-4::par-2].

### Your scores

| `q011-reranked` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 22. `q057-sac`

- Class: **semantic**
- Strategy: `sac`

### Question

Quais informações sobre o tratamento de dados devem ser disponibilizadas ao titular de forma clara e ostensiva, segundo o art. 9º da LGPD?

### Reference answer (golden set)

Devem ser disponibilizadas informações sobre a finalidade específica do tratamento, a forma e duração do tratamento, a identificação do controlador, informações de contato, uso compartilhado de dados, responsabilidades dos agentes, e os direitos do titular (art. 9º, caput).

### Cited authoritative text

**`LEI-13709/2018::art-9::inc-i`**

> I - finalidade específica do tratamento;

**`LEI-13709/2018::art-9::inc-ii`**

> II - forma e duração do tratamento, observados os segredos comercial e industrial;

**`LEI-13709/2018::art-9::inc-iii`**

> III - identificação do controlador;

**`LEI-13709/2018::art-9::inc-iv`**

> IV - informações de contato do controlador;

**`LEI-13709/2018::art-9::inc-v`**

> V - informações acerca do uso compartilhado de dados pelo controlador e a finalidade;

**`LEI-13709/2018::art-9::inc-vi`**

> VI - responsabilidades dos agentes que realizarão o tratamento; e

**`LEI-13709/2018::art-9::inc-vii`**

> VII - direitos do titular, com menção explícita aos direitos contidos no art. 18 desta Lei.

### Generated answer

Segundo o art. 9º da LGPD, o titular tem direito ao acesso facilitado a informações sobre o tratamento de seus dados, que devem ser disponibilizadas de forma clara, adequada e ostensiva, incluindo:

- A finalidade específica do tratamento [LEI-13709/2018::art-9::inc-i];
- A forma e a duração do tratamento, observados os segredos comercial e industrial [LEI-13709/2018::art-9::inc-ii];
- A identificação do controlador [LEI-13709/2018::art-9::inc-iii];
- As informações de contato do controlador [LEI-13709/2018::art-9::inc-iv];
- Informações acerca do uso compartilhado de dados pelo controlador e a finalidade [LEI-13709/2018::art-9::inc-v];
- As responsabilidades dos agentes que realizarão o tratamento [LEI-13709/2018::art-9::inc-vi];
- Os direitos do titular, com menção explícita aos direitos contidos no art. 18 da referida Lei [LEI-13709/2018::art-9::inc-vii].

### Your scores

| `q057-sac` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 23. `q129-contextual`

- Class: **global**
- Strategy: `contextual`

### Question

De modo geral, quais são as etapas do rito processual sancionador estabelecido pela ICVM 607/2019 até a apresentação de defesa?

### Reference answer (golden set)

O processo se inicia com a lavratura de termo de acusação (art. 6º), podendo ser precedido de inquérito administrativo (art. 8º); instaura-se com a citação dos acusados (art. 21); o acusado apresenta defesa em até 30 dias (art. 29); e após a defesa, é designado Relator por sorteio (art. 31).

### Cited authoritative text

**`ICVM-607/2019::art-7`**

> Art. 7º.  Antes da citação dos acusados para apresentação de defesa, a PFE emitirá parecer
> sobre o termo de acusação, no prazo de 30 (trinta) dias contados da data de recebimento do termo
> de acusação, com o seguinte escopo:
> I – exame do cumprimento do art. 5º;
> II – análise objetiva da observância dos requisitos do art. 6º; e
> III – exame da adequação do rito adotado para o processo administrativo sancionador.
> § 1°  Considerando o parecer da PFE, a superintendência tomará as providências que
> considerar cabíveis, podendo, inclusive, arquivar o processo, adequar o rito processual ou realizar
> eventuais ajustes no termo de acusação.
> § 2°  O superintendente deverá justificar a não adoção de eventuais providências
> recomendadas pelo parecer.
> § 3º O parecer da PFE não será obrigatório nos processos administrativos sancionadores
> submetidos ao rito simplificado de que trata o art. 73 desta Instrução.
> § 4º  Sem prejuízo da emissão do parecer de que trata este artigo, as superintendências
> poderão solicitar assessoramento jurídico direto à PFE ainda na fase de instrução.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 6

**`ICVM-607/2019::art-7::par-3`**

> § 3º O parecer da PFE não será obrigatório nos processos administrativos sancionadores
> submetidos ao rito simplificado de que trata o art. 73 desta Instrução.

**`ICVM-607/2019::art-7::par-1`**

> § 1°  Considerando o parecer da PFE, a superintendência tomará as providências que
> considerar cabíveis, podendo, inclusive, arquivar o processo, adequar o rito processual ou realizar
> eventuais ajustes no termo de acusação.

**`ICVM-607/2019::art-21`**

> Art. 21.  Considera-se instaurado o processo administrativo sancionador com a citação dos
> acusados para apresentação de defesa.
> § 1º  A citação conterá:
> I – a identificação do acusado;
> II – a indicação dos fatos imputados ao acusado;
> III – a finalidade da citação;
> IV – o prazo para a apresentação de defesa;
> V – a informação da continuidade do processo, independentemente de seu comparecimento;
> VI – o dever do acusado, ou de procurador por ele constituído, de se cadastrar no sistema de
> processo eletrônico existente na página da CVM na rede mundial de computadores para fins de
> acesso aos autos e posterior acompanhamento do andamento do processo; e
> VII – o aviso de que o acusado poderá propor a celebração de termo de compromisso, em
> conformidade com o disposto no Capítulo IV desta Instrução.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 10
> § 2º  O requisito de que trata o inciso II do § 1º poderá ser atendido por meio da juntada do
> termo ou da peça de acusação.
> § 3º  Quando se tratar de processo em meio físico, a citação deverá conter, além dos requisitos
> previstos no § 1º, o termo de acusação ou a peça acusatória, a depender do caso.
> § 4º  O acusado que, embora citado, não apresentar defesa será considerado revel.

**`ICVM-607/2019::art-21::par-1`**

> § 1º  A citação conterá:
> I – a identificação do acusado;
> II – a indicação dos fatos imputados ao acusado;
> III – a finalidade da citação;
> IV – o prazo para a apresentação de defesa;
> V – a informação da continuidade do processo, independentemente de seu comparecimento;
> VI – o dever do acusado, ou de procurador por ele constituído, de se cadastrar no sistema de
> processo eletrônico existente na página da CVM na rede mundial de computadores para fins de
> acesso aos autos e posterior acompanhamento do andamento do processo; e
> VII – o aviso de que o acusado poderá propor a celebração de termo de compromisso, em
> conformidade com o disposto no Capítulo IV desta Instrução.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 10

**`ICVM-607/2019::art-29`**

> Art. 29.  O acusado deverá apresentar sua defesa por escrito no prazo de 30 (trinta) dias após a
> citação, oportunidade em que deverá juntar os documentos destinados a provar suas alegações e
> especificar as demais provas que pretenda produzir, observado o disposto nos arts. 42 e 43 desta
> Instrução.
> § 1º  O interessado deverá manifestar sua intenção de celebrar termo de compromisso até o
> término do prazo para a apresentação de defesa, e sem prejuízo do ônus de apresentação desta.
> § 2º  A manifestação de intenção ou a apresentação de proposta de termo de compromisso não
> suspende nem interrompe o prazo para apresentação da defesa.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 14
> § 3º  Serão computados individualmente os prazos para todas as manifestações dos acusados,
> sendo admitida uma única prorrogação pelo mesmo prazo, diante de pedido devidamente
> fundamentado.
> § 4º  Os acusados que constituírem o mesmo procurador e apresentarem defesa conjunta terão
> o mesmo prazo para se manifestarem nos autos, contado da citação que for efetivada por último, nos
> termos do art. 29, § 3º, desta Instrução.
> § 5º  Nos processos sancionadores instaurados em desfavor de múltiplos acusados, as defesas
> serão consideradas sigilosas e não serão fornecidas a terceiros ou a outros acusados até o
> encerramento do último prazo de apresentação de defesa.

**`ICVM-607/2019::art-21::par-4`**

> § 4º  O acusado que, embora citado, não apresentar defesa será considerado revel.

### Generated answer

De acordo com a Instrução CVM nº 607/2019, as etapas do rito processual sancionador até a apresentação da defesa são:

1. **Parecer da PFE:** Antes da citação, a Procuradoria Federal Especializada (PFE) deve emitir um parecer sobre o termo de acusação no prazo de 30 dias, examinando o cumprimento de requisitos legais e a adequação do rito [ICVM-607/2019::art-7]. Este parecer não é obrigatório em processos submetidos ao rito simplificado [ICVM-607/2019::art-7::par-3].
2. **Providências da Superintendência:** Com base no parecer da PFE, a superintendência pode arquivar o processo, adequar o rito ou realizar ajustes no termo de acusação [ICVM-607/2019::art-7::par-1].
3. **Citação:** O processo administrativo sancionador é considerado instaurado com a citação dos acusados [ICVM-607/2019::art-21]. A citação deve conter informações como a identificação do acusado, os fatos imputados, o prazo para defesa e o aviso sobre a possibilidade de propor termo de compromisso [ICVM-607/2019::art-21::par-1].
4. **Apresentação de Defesa:** O acusado deve apresentar sua defesa por escrito no prazo de 30 dias após a citação, juntando os documentos probatórios e especificando as provas que pretende produzir [ICVM-607/2019::art-29]. Caso o acusado não apresente defesa, será considerado revel [ICVM-607/2019::art-21::par-4].

### Your scores

| `q129-contextual` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 24. `q190-parent_child`

- Class: **unanswerable**  ·  **UNANSWERABLE**
- Strategy: `parent_child`

### Question

Qual é o percentual médio de propostas de Acordo de Supervisão aceitas pela CVM desde a edição da ICVM 607/2019?

### Reference answer (golden set)

A ICVM 607/2019 estabelece as regras procedimentais para o Acordo de Supervisão, mas não contém dados estatísticos sobre a taxa de aceitação de propostas.

### Cited authoritative text

_The answer cites nothing._

### Generated answer

As informações fornecidas não contêm dados estatísticos ou históricos sobre o percentual de propostas de Acordo de Supervisão aceitas pela CVM desde a edição da Instrução CVM nº 607/2019. O contexto limita-se a descrever os critérios, procedimentos e requisitos formais para a celebração e o julgamento desses acordos.

### Your scores

| `q190-parent_child` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |
| abstention | | |

---

## 25. `q228-sac_contextual`

- Class: **multi_hop**
- Strategy: `sac_contextual`

### Question

Como o adiamento de sessão de julgamento por motivo de força maior (art. 50, § 4º) se relaciona com a exigência de presença mínima de membros do Colegiado (art. 50, caput) na ICVM 607/2019?

### Reference answer (golden set)

O art. 50, caput, exige a presença de no mínimo 3 membros do Colegiado para a sessão se realizar; se isso não ocorrer por motivo de força maior, o art. 50, § 4º, permite que a sessão seja transferida para o primeiro dia útil seguinte, sem necessidade de nova convocação.

### Cited authoritative text

**`ICVM-607/2019::art-50`**

> Art. 50.  A sessão será presidida pelo Presidente da CVM ou, na sua ausência ou
> impedimento, por seu substituto eventual e somente será realizada com a presença de no mínimo 3
> (três) membros do Colegiado.
> § 1º  A participação dos membros do Colegiado nas sessões de julgamento poderá ocorrer por
> videoconferência.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 21
> § 2º  No julgamento dos processos administrativos sancionadores em que não comparecer
> nenhum acusado ou nenhum de seus respectivos procuradores constituídos nos autos, a sessão
> poderá realizar-se por meio eletrônico.
> § 3º  Considerando o andamento dos trabalhos da sessão, fica facultado ao Presidente
> suspendê-la e reiniciá-la no dia útil subsequente, independente de nova convocação e publicação.
> § 4º A sessão que não se realizar, por motivo de força maior, poderá ser transferida para o
> primeiro dia útil seguinte, na hora anteriormente marcada, independentemente de nova convocação
> e publicação.

**`ICVM-607/2019::art-50::par-4`**

> § 4º A sessão que não se realizar, por motivo de força maior, poderá ser transferida para o
> primeiro dia útil seguinte, na hora anteriormente marcada, independentemente de nova convocação
> e publicação.

### Generated answer

A Instrução CVM 607/2019 estabelece que a sessão de julgamento só pode ser realizada com a presença de, no mínimo, 3 (três) membros do Colegiado [ICVM-607/2019::art-50]. Caso a sessão não ocorra por motivo de força maior, ela poderá ser transferida para o primeiro dia útil seguinte, no mesmo horário, sem a necessidade de nova convocação ou publicação [ICVM-607/2019::art-50::par-4].

Embora o texto não explicite uma relação direta de dependência entre o quórum mínimo e o motivo de força maior, a regra do art. 50, § 4º, funciona como uma exceção procedimental que permite a continuidade dos trabalhos em data próxima quando a sessão é impedida de ocorrer, mantendo-se a validade da pauta original sem a necessidade de novos trâmites de convocação [ICVM-607/2019::art-50::par-4].

### Your scores

| `q228-sac_contextual` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 26. `q011-dense`

- Class: **section_comparative**
- Strategy: `dense`

### Question

Compare as regras de fornecimento de informações sigilosas ao Poder Judiciário (art. 3º) e ao Poder Legislativo Federal (art. 4º) na LC 105/2001.

### Reference answer (golden set)

Ao Poder Judiciário, as informações ordenadas são prestadas preservando o caráter sigiloso mediante acesso restrito às partes do processo (art. 3º). Ao Poder Legislativo Federal, as informações e documentos sigilosos são fornecidos quando fundamentadamente necessários ao exercício de competências constitucionais e legais, sendo que, no caso de comissões parlamentares de inquérito, as solicitações devem ser previamente aprovadas pelo plenário da respectiva Casa ou comissão (art. 4º).

### Cited authoritative text

**`LC-105/2001::art-3`**

> Art. 3o Serão prestadas pelo Banco Central do Brasil, pela Comissão de Valores Mobiliários e pelas instituições financeiras as informações ordenadas pelo Poder Judiciário, preservado o seu caráter sigiloso mediante acesso restrito às partes, que delas não poderão servir-se para fins estranhos à lide.
> § 1o Dependem de prévia autorização do Poder Judiciário a prestação de informações e o fornecimento de documentos sigilosos solicitados por comissão de inquérito administrativo destinada a apurar responsabilidade de servidor público por infração praticada no exercício de suas atribuições, ou que tenha relação com as atribuições do cargo em que se encontre investido.
> § 2o Nas hipóteses do § 1o, o requerimento de quebra de sigilo independe da existência de processo judicial em curso.
> § 3o Além dos casos previstos neste artigo o Banco Central do Brasil e a Comissão de Valores Mobiliários fornecerão à Advocacia-Geral da União as informações e os documentos necessários à defesa da União nas ações em que seja parte.

**`LC-105/2001::art-4`**

> Art. 4o O Banco Central do Brasil e a Comissão de Valores Mobiliários, nas áreas de suas atribuições, e as instituições financeiras fornecerão ao Poder Legislativo Federal as informações e os documentos sigilosos que, fundamentadamente, se fizerem necessários ao exercício de suas respectivas competências constitucionais e legais.
> § 1o As comissões parlamentares de inquérito, no exercício de sua competência constitucional e legal de ampla investigação, obterão as informações e documentos sigilosos de que necessitarem, diretamente das instituições financeiras, ou por intermédio do Banco Central do Brasil ou da Comissão de Valores Mobiliários.
> § 2o As solicitações de que trata este artigo deverão ser previamente aprovadas pelo Plenário da Câmara dos Deputados, do Senado Federal, ou do plenário de suas respectivas comissões parlamentares de inquérito.

**`LC-105/2001::art-4::par-1`**

> § 1o As comissões parlamentares de inquérito, no exercício de sua competência constitucional e legal de ampla investigação, obterão as informações e documentos sigilosos de que necessitarem, diretamente das instituições financeiras, ou por intermédio do Banco Central do Brasil ou da Comissão de Valores Mobiliários.

**`LC-105/2001::art-4::par-2`**

> § 2o As solicitações de que trata este artigo deverão ser previamente aprovadas pelo Plenário da Câmara dos Deputados, do Senado Federal, ou do plenário de suas respectivas comissões parlamentares de inquérito.

### Generated answer

As regras para o fornecimento de informações sigilosas pelo Banco Central do Brasil, pela Comissão de Valores Mobiliários (CVM) e pelas instituições financeiras diferem conforme o Poder solicitante, conforme estabelecido na Lei Complementar nº 105/2001:

**Fornecimento ao Poder Judiciário (Art. 3º):**

- **Obrigatoriedade:** As instituições devem prestar as informações ordenadas pelo Poder Judiciário [LC-105/2001::art-3].
- **Preservação do sigilo:** O caráter sigiloso deve ser preservado mediante acesso restrito às partes, que não podem utilizar as informações para fins estranhos à lide [LC-105/2001::art-3].
- **Comissões de inquérito administrativo:** Caso a solicitação venha de uma comissão de inquérito administrativo (para apurar responsabilidade de servidor público), é necessária a prévia autorização do Poder Judiciário, independentemente da existência de processo judicial em curso [LC-105/2001::art-3::par-1, LC-105/2001::art-3::par-2].

**Fornecimento ao Poder Legislativo Federal (Art. 4º):**

- **Fundamentação:** As informações e documentos sigilosos devem ser fornecidos quando, fundamentadamente, se fizerem necessários ao exercício das competências constitucionais e legais do Poder Legislativo [LC-105/2001::art-4].
- **Comissões Parlamentares de Inquérito (CPIs):** As CPIs podem obter as informações diretamente das instituições financeiras ou por intermédio do Banco Central ou da CVM, no exercício de sua competência de ampla investigação [LC-105/2001::art-4::par-1].
- **Aprovação prévia:** Diferente do Judiciário, as solicitações feitas ao Legislativo exigem aprovação prévia pelo Plenário da Câmara dos Deputados, do Senado Federal ou pelo plenário das respectivas comissões parlamentares de inquérito [LC-105/2001::art-4::par-2].

### Your scores

| `q011-dense` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 27. `q129-raptor`

- Class: **global**
- Strategy: `raptor`

### Question

De modo geral, quais são as etapas do rito processual sancionador estabelecido pela ICVM 607/2019 até a apresentação de defesa?

### Reference answer (golden set)

O processo se inicia com a lavratura de termo de acusação (art. 6º), podendo ser precedido de inquérito administrativo (art. 8º); instaura-se com a citação dos acusados (art. 21); o acusado apresenta defesa em até 30 dias (art. 29); e após a defesa, é designado Relator por sorteio (art. 31).

### Cited authoritative text

**`ICVM-607/2019::art-5`**

> Art. 5º  Previamente à formulação da acusação, as superintendências deverão diligenciar no
> sentido de obter diretamente do investigado esclarecimentos sobre os fatos que podem ser a ele
> imputados.
> Parágrafo único.  Considera-se atendido o disposto no caput sempre que o investigado:
> I – tenha prestado depoimento pessoal ou se manifestado voluntariamente acerca dos fatos
> que podem ser a ele imputados; ou
> II – tenha sido oficiado para prestar esclarecimentos sobre os fatos que podem ser a ele
> imputados, ainda que não o faça.

**`ICVM-607/2019::art-7`**

> Art. 7º.  Antes da citação dos acusados para apresentação de defesa, a PFE emitirá parecer
> sobre o termo de acusação, no prazo de 30 (trinta) dias contados da data de recebimento do termo
> de acusação, com o seguinte escopo:
> I – exame do cumprimento do art. 5º;
> II – análise objetiva da observância dos requisitos do art. 6º; e
> III – exame da adequação do rito adotado para o processo administrativo sancionador.
> § 1°  Considerando o parecer da PFE, a superintendência tomará as providências que
> considerar cabíveis, podendo, inclusive, arquivar o processo, adequar o rito processual ou realizar
> eventuais ajustes no termo de acusação.
> § 2°  O superintendente deverá justificar a não adoção de eventuais providências
> recomendadas pelo parecer.
> § 3º O parecer da PFE não será obrigatório nos processos administrativos sancionadores
> submetidos ao rito simplificado de que trata o art. 73 desta Instrução.
> § 4º  Sem prejuízo da emissão do parecer de que trata este artigo, as superintendências
> poderão solicitar assessoramento jurídico direto à PFE ainda na fase de instrução.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 6

**`ICVM-607/2019::art-21`**

> Art. 21.  Considera-se instaurado o processo administrativo sancionador com a citação dos
> acusados para apresentação de defesa.
> § 1º  A citação conterá:
> I – a identificação do acusado;
> II – a indicação dos fatos imputados ao acusado;
> III – a finalidade da citação;
> IV – o prazo para a apresentação de defesa;
> V – a informação da continuidade do processo, independentemente de seu comparecimento;
> VI – o dever do acusado, ou de procurador por ele constituído, de se cadastrar no sistema de
> processo eletrônico existente na página da CVM na rede mundial de computadores para fins de
> acesso aos autos e posterior acompanhamento do andamento do processo; e
> VII – o aviso de que o acusado poderá propor a celebração de termo de compromisso, em
> conformidade com o disposto no Capítulo IV desta Instrução.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 10
> § 2º  O requisito de que trata o inciso II do § 1º poderá ser atendido por meio da juntada do
> termo ou da peça de acusação.
> § 3º  Quando se tratar de processo em meio físico, a citação deverá conter, além dos requisitos
> previstos no § 1º, o termo de acusação ou a peça acusatória, a depender do caso.
> § 4º  O acusado que, embora citado, não apresentar defesa será considerado revel.

**`ICVM-607/2019::art-21, par-1`**

> _(not resolvable - the cited ID does not exist in the parsed corpus)_

**`ICVM-607/2019::art-21, ICVM-607/2019::art-23`**

> _(not resolvable - the cited ID does not exist in the parsed corpus)_

**`ICVM-607/2019::art-26, ICVM-607/2019::art-30`**

> _(not resolvable - the cited ID does not exist in the parsed corpus)_

### Generated answer

De acordo com a Instrução CVM nº 607/2019, o rito processual sancionador até a apresentação da defesa segue, de modo geral, as seguintes etapas:

1. **Investigação e Esclarecimentos Prévios:** As superintendências investigam as condutas e podem buscar esclarecimentos prévios do investigado [ICVM-607/2019::art-5].
2. **Decisão sobre o Procedimento:** Com base na avaliação da conduta (considerando critérios como boa-fé, ressarcimento e impacto no mercado), a superintendência decide entre a lavratura de termo de acusação, a proposta de inquérito administrativo ou a adoção de medidas de supervisão [ICVM-607/2019::art-4, ICVM-607/2019::art-6, ICVM-607/2019::art-8].
3. **Submissão à PFE:** Havendo elementos de autoria e materialidade, o termo de acusação é submetido à Procuradoria Federal Especializada (PFE) [ICVM-607/2019::art-7].
4. **Instauração e Citação:** O processo administrativo sancionador é instaurado com a citação do acusado [ICVM-607/2019::art-21]. A citação deve conter a identificação, os fatos imputados, o prazo de defesa, o dever de cadastro no sistema eletrônico e o aviso sobre a possibilidade de termo de compromisso [ICVM-607/2019::art-21, par-1].
5. **Comunicação:** A citação é realizada preferencialmente por meio eletrônico, utilizando-se a via postal ou edital no "Diário Eletrônico" de forma subsidiária [ICVM-607/2019::art-21, ICVM-607/2019::art-23].
6. **Apresentação de Defesa:** O acusado possui o prazo de 30 dias úteis para apresentar sua defesa escrita, sendo obrigatória a regularização da representação processual em até 15 dias, sob pena de desconsideração da peça defensiva [ICVM-607/2019::art-26, ICVM-607/2019::art-30].

### Your scores

| `q129-raptor` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 28. `q017-raptor`

- Class: **unanswerable**  ·  **UNANSWERABLE**
- Strategy: `raptor`

### Question

Qual é o valor da multa pecuniária prevista na RES-CMN-5274/2025 para o descumprimento do prazo de adaptação até março de 2026?

### Reference answer (golden set)

A RES-CMN-5274/2025 não estabelece um valor de multa pecuniária para o descumprimento desse prazo; o texto da resolução não trata dessa questão.

### Cited authoritative text

**`RES-CMN-5274/2025::art-2`**

> Art. 2º As instituições em funcionamento na data da entrada em vigor desta Resolução devem promover as adaptações necessárias à adequação ao disposto nesta Resolução até 1º de março de 2026.

### Generated answer

A Resolução CMN nº 5.274/2025 não contém informações sobre o valor de multas pecuniárias para o descumprimento do prazo de adaptação. O texto normativo estabelece apenas que as instituições em funcionamento devem promover as adaptações necessárias até 1º de março de 2026 [RES-CMN-5274/2025::art-2].

### Your scores

| `q017-raptor` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |
| abstention | | |

---

## 29. `q024-contextual`

- Class: **exact_factual**
- Strategy: `contextual`

### Question

Quem deve aprovar a política de segurança cibernética e o plano de ação e resposta a incidentes, segundo o art. 9º da RES-CMN-4893/2021?

### Reference answer (golden set)

Devem ser aprovados pelo conselho de administração ou, na sua inexistência, pela diretoria da instituição (art. 9º).

### Cited authoritative text

**`RES-CMN-4893/2021::art-9`**

> Art. 9º  A política de segurança cibernética referida no art. 2º e o plano de ação e de resposta a incidentes mencionado no art. 6º devem ser
> aprovados pelo conselho de administração ou, na sua inexistência, pela diretoria da instituição.

### Generated answer

Segundo o art. 9º da Resolução CMN nº 4.893/2021, a política de segurança cibernética e o plano de ação e de resposta a incidentes devem ser aprovados pelo conselho de administração ou, na sua inexistência, pela diretoria da instituição [RES-CMN-4893/2021::art-9].

### Your scores

| `q024-contextual` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 30. `q193-raptor`

- Class: **exact_factual**
- Strategy: `raptor`

### Question

A participação do investidor lesado na análise da proposta de termo de compromisso lhe confere a condição de parte no processo, segundo o art. 85, § 1º da ICVM 607/2019?

### Reference answer (golden set)

Não, a participação do investidor lesado não lhe confere a condição de parte no processo administrativo (art. 85, § 1º).

### Cited authoritative text

**`ICVM-607/2019::art-85::par-1`**

> § 1º  A participação do investidor lesado não lhe confere a condição de parte no processo
> administrativo.

### Generated answer

Não. Segundo o art. 85, § 1º, da Instrução CVM nº 607/2019, a participação do investidor lesado na análise da proposta de termo de compromisso não lhe confere a condição de parte no processo administrativo [ICVM-607/2019::art-85::par-1].

### Your scores

| `q193-raptor` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 31. `q123-hybrid_rrf`

- Class: **numeric_tabular**
- Strategy: `hybrid_rrf`

### Question

Em quantos dias o instrumento de mandato de um procurador não constituído nos autos deve ser apresentado à CVM, segundo o art. 30, § 1º da ICVM 607/2019?

### Reference answer (golden set)

Nos 15 (quinze) dias subsequentes à apresentação da defesa (art. 30, § 1º).

### Cited authoritative text

**`ICVM-607/2019::art-30::par-1`**

> § 1º  Será admitida defesa firmada por procurador que não esteja constituído nos autos desde
> que o respectivo instrumento de mandato seja apresentado à CVM nos 15 (quinze) dias
> subsequentes à apresentação da defesa.

### Generated answer

Segundo o art. 30, § 1º da Instrução CVM nº 607/2019, o instrumento de mandato deve ser apresentado à CVM nos 15 (quinze) dias subsequentes à apresentação da defesa [ICVM-607/2019::art-30::par-1].

### Your scores

| `q123-hybrid_rrf` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 32. `q205-sac_contextual`

- Class: **numeric_tabular**
- Strategy: `sac_contextual`

### Question

De quanto é o benefício exclusivo para o signatário que não for o primeiro a se qualificar em relação à infração noticiada, segundo o art. 107, § 1º da ICVM 607/2019?

### Reference answer (golden set)

Poderá beneficiar-se exclusivamente da redução de 1/3 (um terço) da penalidade a ela aplicável (art. 107, § 1º).

### Cited authoritative text

**`ICVM-607/2019::art-107::par-1`**

> § 1º  A pessoa natural ou jurídica que não for a primeira a se qualificar com respeito à
> infração noticiada ou sob investigação poderá beneficiar-se exclusivamente da redução de 1/3 (um
> terço) da penalidade a ela aplicável.

### Generated answer

De acordo com o art. 107, § 1º da Instrução CVM nº 607/2019, a pessoa natural ou jurídica que não for a primeira a se qualificar com respeito à infração noticiada ou sob investigação pode beneficiar-se exclusivamente da redução de 1/3 (um terço) da penalidade a ela aplicável [ICVM-607/2019::art-107::par-1].

### Your scores

| `q205-sac_contextual` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 33. `q093-hybrid_rrf`

- Class: **semantic**
- Strategy: `hybrid_rrf`

### Question

O que o controlador deve levar em consideração ao estabelecer regras de boas práticas, segundo o art. 50, § 1º da LGPD?

### Reference answer (golden set)

Deve levar em consideração, em relação ao tratamento e aos dados, a natureza, o escopo, a finalidade e a probabilidade e a gravidade dos riscos e dos benefícios decorrentes do tratamento de dados do titular (art. 50, § 1º).

### Cited authoritative text

**`LEI-13709/2018::art-50::par-1`**

> § 1º Ao estabelecer regras de boas práticas, o controlador e o operador levarão em consideração, em relação ao tratamento e aos dados, a natureza, o escopo, a finalidade e a probabilidade e a gravidade dos riscos e dos benefícios decorrentes de tratamento de dados do titular.

### Generated answer

Ao estabelecer regras de boas práticas, o controlador deve levar em consideração, em relação ao tratamento e aos dados, a natureza, o escopo, a finalidade e a probabilidade e a gravidade dos riscos e dos benefícios decorrentes do tratamento de dados do titular [LEI-13709/2018::art-50::par-1].

### Your scores

| `q093-hybrid_rrf` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 34. `q187-graphrag`

- Class: **multi_hop**
- Strategy: `graphrag`

### Question

Como o prazo de manifestação de intenção de celebrar termo de compromisso do art. 82, § 1º, se relaciona com o prazo de apresentação de defesa do art. 29 da ICVM 607/2019?

### Reference answer (golden set)

O art. 29 estabelece o prazo de 30 dias após a citação para apresentação de defesa; o art. 82, § 1º, exige que a intenção de celebrar termo de compromisso seja manifestada dentro desse mesmo prazo de defesa.

### Cited authoritative text

_The answer cites nothing._

### Generated answer

O interessado deve manifestar sua intenção de celebrar o termo de compromisso dentro do prazo estabelecido para a apresentação de defesa [ICVM-607/2019::art-82::par-1, ICVM-607/2019::art-29::par-1].

Essa manifestação de intenção, ou mesmo a apresentação da proposta de termo de compromisso, não suspende nem interrompe o prazo de 30 dias para a apresentação da defesa, sendo que o interessado mantém o ônus de apresentá-la [ICVM-607/2019::art-29, ICVM-607/2019::art-29::par-1, ICVM-607/2019::art-29::par-2].

### Your scores

| `q187-graphrag` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 35. `q128-hybrid_rrf`

- Class: **section_comparative**
- Strategy: `hybrid_rrf`

### Question

Compare o prazo para conclusão do inquérito administrativo (art. 10) com o prazo para apresentação de defesa (art. 29) na ICVM 607/2019.

### Reference answer (golden set)

O inquérito administrativo deve ser concluído em 120 dias, prorrogáveis mediante pedido motivado (art. 10); já a apresentação de defesa pelo acusado, após a citação no processo sancionador, tem prazo de apenas 30 dias (art. 29) - um prazo bem mais curto, pois já parte de acusação formalizada.

### Cited authoritative text

**`ICVM-607/2019::art-10`**

> Art. 10.  Os trabalhos de investigação deverão ser concluídos em 120 (cento e vinte) dias
> contados da data de instauração do inquérito administrativo, podendo tal prazo ser prorrogado, mais
> de uma vez, por meio de pedido motivado encaminhado à Superintendência Geral, com indicação
> de novo prazo.
> Parágrafo único.  Caberá à Superintendência Geral, com base na motivação que lhe for
> apresentada, apreciar o pedido de prorrogação de prazo, podendo, em sendo o caso, fixar prazo
> inferior ao solicitado.

**`ICVM-607/2019::art-29`**

> Art. 29.  O acusado deverá apresentar sua defesa por escrito no prazo de 30 (trinta) dias após a
> citação, oportunidade em que deverá juntar os documentos destinados a provar suas alegações e
> especificar as demais provas que pretenda produzir, observado o disposto nos arts. 42 e 43 desta
> Instrução.
> § 1º  O interessado deverá manifestar sua intenção de celebrar termo de compromisso até o
> término do prazo para a apresentação de defesa, e sem prejuízo do ônus de apresentação desta.
> § 2º  A manifestação de intenção ou a apresentação de proposta de termo de compromisso não
> suspende nem interrompe o prazo para apresentação da defesa.
> COMISSÃO DE VALORES MOBILIÁRIOS
> Rua Sete de Setembro, 111/2-5º e 23-34º Andares, Centro, Rio de Janeiro/RJ – CEP: 20050-901 – Brasil - Tel.: (21) 3554-8686
> Rua Cincinato Braga, 340/2º, 3º e 4º Andares, Bela Vista, São Paulo/ SP – CEP: 01333-010 – Brasil - Tel.: (11) 2146-2000
> SCN Q.02 – Bl. A – Ed. Corporate Financial Center, S.404/4º Andar, Brasília/DF – CEP: 70712-900 – Brasil -Tel.: (61) 3327-2030/2031
> <www.cvm.gov.br>
> INSTRUÇÃO CVM Nº 607, DE 17 DE JUNHO DE 2019
> 14
> § 3º  Serão computados individualmente os prazos para todas as manifestações dos acusados,
> sendo admitida uma única prorrogação pelo mesmo prazo, diante de pedido devidamente
> fundamentado.
> § 4º  Os acusados que constituírem o mesmo procurador e apresentarem defesa conjunta terão
> o mesmo prazo para se manifestarem nos autos, contado da citação que for efetivada por último, nos
> termos do art. 29, § 3º, desta Instrução.
> § 5º  Nos processos sancionadores instaurados em desfavor de múltiplos acusados, as defesas
> serão consideradas sigilosas e não serão fornecidas a terceiros ou a outros acusados até o
> encerramento do último prazo de apresentação de defesa.

**`ICVM-607/2019::art-29::par-3`**

> § 3º  Serão computados individualmente os prazos para todas as manifestações dos acusados,
> sendo admitida uma única prorrogação pelo mesmo prazo, diante de pedido devidamente
> fundamentado.

### Generated answer

De acordo com a Instrução CVM nº 607/2019, os prazos comparados são:

- **Inquérito Administrativo:** Os trabalhos de investigação devem ser concluídos no prazo de 120 dias, contados a partir da data de instauração do inquérito [ICVM-607/2019::art-10]. Este prazo pode ser prorrogado mais de uma vez, mediante pedido motivado encaminhado à Superintendência Geral [ICVM-607/2019::art-10].
- **Apresentação de Defesa:** O acusado possui o prazo de 30 dias após a citação para apresentar sua defesa por escrito [ICVM-607/2019::art-29]. Este prazo é computado individualmente para cada acusado e admite uma única prorrogação pelo mesmo período, desde que haja pedido devidamente fundamentado [ICVM-607/2019::art-29::par-3].

### Your scores

| `q128-hybrid_rrf` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---

## 36. `q107-dense`

- Class: **semantic**
- Strategy: `dense`

### Question

O que é considerado 'dado anonimizado' para os fins da LGPD, segundo o art. 5º, inciso III?

### Reference answer (golden set)

É o dado relativo a um titular que não possa ser identificado, considerando a utilização de meios técnicos razoáveis e disponíveis na ocasião de seu tratamento (art. 5º, III).

### Cited authoritative text

**`LEI-13709/2018::art-5::inc-iii`**

> III - dado anonimizado: dado relativo a titular que não possa ser identificado, considerando a utilização de meios técnicos razoáveis e disponíveis na ocasião de seu tratamento;

### Generated answer

Segundo o art. 5º, inciso III, da LGPD, dado anonimizado é aquele relativo a um titular que não possa ser identificado, considerando a utilização de meios técnicos razoáveis e disponíveis na ocasião de seu tratamento [LEI-13709/2018::art-5::inc-iii].

### Your scores

| `q107-dense` | 0.0 / 0.5 / 1.0 | notes |
| --- | --- | --- |
| faithfulness | | |
| answer_relevancy | | |

---
