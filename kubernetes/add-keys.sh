for f in mini.t2t.job.yaml baseline.t2t.job.yaml
do
  yq w -i -d1 $f \
    "spec.template.spec.containers[0].env.(name==AWS_ACCESS_KEY_ID).value" \
    $AWS_ACCESS_KEY_ID

  yq w -i -d1 $f \
    "spec.template.spec.containers[0].env.(name==AWS_SECRET_ACCESS_KEY).value" \
    $AWS_SECRET_ACCESS_KEY
done




