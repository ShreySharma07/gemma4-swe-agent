## fastapi_13207
necessary groups: ['fastapi/_compat/v2.py::get_definitions', 'fastapi/_compat/v2.py::get_schema_from_model_field']

### fastapi/_compat/v2.py::get_definitions
```diff
@@ -203,9 +208,14 @@ def get_definitions(
     Dict[Tuple[ModelField, Literal["validation", "serialization"]], JsonSchemaValue],
     Dict[str, Dict[str, Any]],
 ]:
+    has_computed_fields: bool = any(
+        field._type_adapter.core_schema.get("schema", {}).get("computed_fields", [])
+        for field in fields
+    )
+
     schema_generator = GenerateJsonSchema(ref_template=REF_TEMPLATE)
     override_mode: Union[Literal["validation"], None] = (
-        None if separate_input_output_schemas else "validation"
+        None if (separate_input_output_schemas or has_computed_fields) else "validation"
     )
     validation_fields = [field for field in fields if field.mode == "validation"]
     serialization_fields = [field for field in fields if field.mode == "serialization"]```

### fastapi/_compat/v2.py::get_schema_from_model_field
```diff
@@ -180,8 +180,13 @@ def get_schema_from_model_field(
     ],
     separate_input_output_schemas: bool = True,
 ) -> Dict[str, Any]:
+    computed_fields = field._type_adapter.core_schema.get("schema", {}).get(
+        "computed_fields", []
+    )
     override_mode: Union[Literal["validation"], None] = (
-        None if separate_input_output_schemas else "validation"
+        None
+        if (separate_input_output_schemas or len(computed_fields) > 0)
+        else "validation"
     )
     # This expects that GenerateJsonSchema was already used to generate the definitions
     json_schema = field_mapping[(field, override_mode or field.mode)]
```

## fastapi_13713
necessary groups: ['fastapi/applications.py::FastAPI.__init__', 'fastapi/applications.py::FastAPI.openapi', 'fastapi/openapi/utils.py::get_openapi']

### fastapi/applications.py::FastAPI.__init__
```diff
@@ -810,6 +810,32 @@ class Item(BaseModel):
                 """
             ),
         ] = True,
+        openapi_external_docs: Annotated[
+            Optional[Dict[str, Any]],
+            Doc(
+                """
+                This field allows you to provide additional external documentation links.
+                If provided, it must be a dictionary containing:
+
+                * `description`: A brief description of the external documentation.
+                * `url`: The URL pointing to the external documentation. The value **MUST**
+                be a valid URL format.
+
+                **Example**:
+
+                ```python
+                from fastapi import FastAPI
+
+                external_docs = {
+                    "description": "Detailed API Reference",
+                    "url": "https://example.com/api-docs",
+                }
+
+                app = FastAPI(openapi_external_docs=external_docs)
+                ```
+                """
+            ),
+        ] = None,
         **extra: Annotated[
             Any,
             Doc(
@@ -838,6 +864,7 @@ class Item(BaseModel):
         self.swagger_ui_parameters = swagger_ui_parameters
         self.servers = servers or []
         self.separate_input_output_schemas = separate_input_output_schemas
+        self.openapi_external_docs = openapi_external_docs
         self.extra = extra
         self.openapi_version: Annotated[
```

### fastapi/applications.py::FastAPI.openapi
```diff
@@ -992,6 +1019,7 @@ def openapi(self) -> Dict[str, Any]:
                 tags=self.openapi_tags,
                 servers=self.servers,
                 separate_input_output_schemas=self.separate_input_output_schemas,
+                external_docs=self.openapi_external_docs,
             )
         return self.openapi_schema
 
```

### fastapi/openapi/utils.py::get_openapi
```diff
@@ -488,6 +488,7 @@ def get_openapi(
     contact: Optional[Dict[str, Union[str, Any]]] = None,
     license_info: Optional[Dict[str, Union[str, Any]]] = None,
     separate_input_output_schemas: bool = True,
+    external_docs: Optional[Dict[str, Any]] = None,
 ) -> Dict[str, Any]:
     info: Dict[str, Any] = {"title": title, "version": version}
     if summary:
@@ -565,4 +566,6 @@ def get_openapi(
         output["webhooks"] = webhook_paths
     if tags:
         output["tags"] = tags
+    if external_docs:
+        output["externalDocs"] = external_docs
     return jsonable_encoder(OpenAPI(**output), by_alias=True, exclude_none=True)  # type: ignore```

## fastapi_13786
necessary groups: ['fastapi/security/api_key.py::APIKeyCookie.__call__ | fastapi/security/api_key.py::APIKeyCookie.__init__', 'fastapi/security/api_key.py::APIKeyHeader.__call__ | fastapi/security/api_key.py::APIKeyHeader.__init__', 'fastapi/security/api_key.py::APIKeyQuery.__call__ | fastapi/security/api_key.py::APIKeyQuery.__init__', 'fastapi/security/http.py::HTTPBearer.__call__', 'fastapi/security/http.py::HTTPDigest.__call__']

### fastapi/security/api_key.py::APIKeyCookie.__call__ | fastapi/security/api_key.py::APIKeyCookie.__init__
```diff
@@ -276,14 +306,14 @@ def __init__(
             ),
         ] = True,
     ):
-        self.model: APIKey = APIKey(
-            **{"in": APIKeyIn.cookie},
+        super().__init__(
+            location=APIKeyIn.cookie,
             name=name,
+            scheme_name=scheme_name,
             description=description,
+            auto_error=auto_error,
         )
-        self.scheme_name = scheme_name or self.__class__.__name__
-        self.auto_error = auto_error
 
     async def __call__(self, request: Request) -> Optional[str]:
         api_key = request.cookies.get(self.model.name)
-        return self.check_api_key(api_key, self.auto_error)
+        return self.check_api_key(api_key)
```

### fastapi/security/api_key.py::APIKeyHeader.__call__ | fastapi/security/api_key.py::APIKeyHeader.__init__
```diff
@@ -188,17 +218,17 @@ def __init__(
             ),
         ] = True,
     ):
-        self.model: APIKey = APIKey(
-            **{"in": APIKeyIn.header},
+        super().__init__(
+            location=APIKeyIn.header,
             name=name,
+            scheme_name=scheme_name,
             description=description,
+            auto_error=auto_error,
         )
-        self.scheme_name = scheme_name or self.__class__.__name__
-        self.auto_error = auto_error
 
     async def __call__(self, request: Request) -> Optional[str]:
         api_key = request.headers.get(self.model.name)
-        return self.check_api_key(api_key, self.auto_error)
+        return self.check_api_key(api_key)
 
 
 class APIKeyCookie(APIKeyBase):
```

### fastapi/security/api_key.py::APIKeyQuery.__call__ | fastapi/security/api_key.py::APIKeyQuery.__init__
```diff
@@ -100,17 +130,17 @@ def __init__(
             ),
         ] = True,
     ):
-        self.model: APIKey = APIKey(
-            **{"in": APIKeyIn.query},
+        super().__init__(
+            location=APIKeyIn.query,
             name=name,
+            scheme_name=scheme_name,
             description=description,
+            auto_error=auto_error,
         )
-        self.scheme_name = scheme_name or self.__class__.__name__
-        self.auto_error = auto_error
 
     async def __call__(self, request: Request) -> Optional[str]:
         api_key = request.query_params.get(self.model.name)
-        return self.check_api_key(api_key, self.auto_error)
+        return self.check_api_key(api_key)
 
 
 class APIKeyHeader(APIKeyBase):
```

### fastapi/security/http.py::HTTPBearer.__call__
```diff
@@ -306,17 +310,12 @@ async def __call__(
         scheme, credentials = get_authorization_scheme_param(authorization)
         if not (authorization and scheme and credentials):
             if self.auto_error:
-                raise HTTPException(
-                    status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
-                )
+                raise self.make_not_authenticated_error()
             else:
                 return None
         if scheme.lower() != "bearer":
             if self.auto_error:
-                raise HTTPException(
-                    status_code=HTTP_403_FORBIDDEN,
-                    detail="Invalid authentication credentials",
-                )
+                raise self.make_not_authenticated_error()
             else:
                 return None
         return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)
```

### fastapi/security/http.py::HTTPDigest.__call__
```diff
@@ -408,17 +413,12 @@ async def __call__(
         scheme, credentials = get_authorization_scheme_param(authorization)
         if not (authorization and scheme and credentials):
             if self.auto_error:
-                raise HTTPException(
-                    status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
-                )
+                raise self.make_not_authenticated_error()
             else:
                 return None
         if scheme.lower() != "digest":
             if self.auto_error:
-                raise HTTPException(
-                    status_code=HTTP_403_FORBIDDEN,
-                    detail="Invalid authentication credentials",
-                )
+                raise self.make_not_authenticated_error()
             else:
                 return None
         return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)
```

## fastapi_14262
necessary groups: ['fastapi/dependencies/utils.py::get_dependant', 'fastapi/dependencies/utils.py::solve_dependencies', 'fastapi/param_functions.py::Depends', 'fastapi/routing.py::request_response.<locals>.app.<locals>.app', 'fastapi/routing.py::websocket_session.<locals>.app.<locals>.app']

### fastapi/dependencies/utils.py::get_dependant
```diff
@@ -237,21 +230,23 @@ def get_dependant(
     name: Optional[str] = None,
     security_scopes: Optional[List[str]] = None,
     use_cache: bool = True,
+    scope: Union[Literal["function", "request"], None] = None,
 ) -> Dependant:
     dependant = Dependant(
         call=call,
         name=name,
         path=path,
         security_scopes=security_scopes,
         use_cache=use_cache,
+        scope=scope,
     )
     path_param_names = get_path_param_names(path)
     endpoint_signature = get_typed_signature(call)
     signature_params = endpoint_signature.parameters
     if isinstance(call, SecurityBase):
         use_scopes: List[str] = []
         if isinstance(call, (OAuth2, OpenIdConnect)):
-            use_scopes = security_scopes
+            use_scopes = security_scopes or use_scopes
         security_requirement = SecurityRequirement(
             security_scheme=call, scopes=use_scopes
         )
@@ -266,6 +261,16 @@ def get_dependant(
         )
         if param_details.depends is not None:
             assert param_details.depends.dependency
+            if (
+                (dependant.is_gen_callable or dependant.is_async_gen_callable)
+                and dependant.computed_scope == "request"
+                and param_details.depends.scope == "function"
+            ):
+                assert dependant.call
+                raise DependencyScopeError(
+                    f'The dependency "{dependant.call.__name__}" has a scope of '
+                    '"request", it cannot depend on dependencies with scope "function".'
+                )
             use_security_scopes = security_scopes or []
```

### fastapi/dependencies/utils.py::solve_dependencies
```diff
@@ -582,10 +566,20 @@ async def solve_dependencies(
     background_tasks: Optional[StarletteBackgroundTasks] = None,
     response: Optional[Response] = None,
     dependency_overrides_provider: Optional[Any] = None,
-    dependency_cache: Optional[Dict[Tuple[Callable[..., Any], Tuple[str]], Any]] = None,
+    dependency_cache: Optional[Dict[DependencyCacheKey, Any]] = None,
+    # TODO: remove this parameter later, no longer used, not removing it yet as some
+    # people might be monkey patching this function (although that's not supported)
     async_exit_stack: AsyncExitStack,
     embed_body_fields: bool,
 ) -> SolvedDependency:
+    request_astack = request.scope.get("fastapi_inner_astack")
+    assert isinstance(request_astack, AsyncExitStack), (
+        "fastapi_inner_astack not found in request scope"
+    )
+    function_astack = request.scope.get("fastapi_function_astack")
+    assert isinstance(function_astack, AsyncExitStack), (
+        "fastapi_function_astack not found in request scope"
+    )
     values: Dict[str, Any] = {}
     errors: List[Any] = []
     if response is None:
@@ -594,12 +588,8 @@ async def solve_dependencies(
         response.status_code = None  # type: ignore
     if dependency_cache is None:
         dependency_cache = {}
-    sub_dependant: Dependant
     for sub_dependant in dependant.dependencies:
         sub_dependant.call = cast(Callable[..., Any], sub_dependant.call)
-        sub_dependant.cache_key = cast(
-            Tuple[Callable[..., Any], Tuple[str]], sub_dependant.cache_key
-        )
         call = sub_dependant.call
         use_sub_dependant = sub_dependant
         if (
@@ -616,6 +606,7 @@ async def solve_dependencies(
                 call=call,
                 name=sub_dependant.name,
                 security_scopes=sub_dependant.security_scopes,
+                scope=sub_dependant.scope,
```

### fastapi/param_functions.py::Depends
```diff
@@ -2245,6 +2245,26 @@ def Depends(  # noqa: N802
             """
         ),
     ] = True,
+    scope: Annotated[
+        Union[Literal["function", "request"], None],
+        Doc(
+            """
+            Mainly for dependencies with `yield`, define when the dependency function
+            should start (the code before `yield`) and when it should end (the code
+            after `yield`).
+
+            * `"function"`: start the dependency before the *path operation function*
+                that handles the request, end the dependency after the *path operation
+                function* ends, but **before** the response is sent back to the client.
+                So, the dependency function will be executed **around** the *path operation
+                **function***.
+            * `"request"`: start the dependency before the *path operation function*
+                that handles the request (similar to when using `"function"`), but end
+                **after** the response is sent back to the client. So, the dependency
+                function will be executed **around** the **request** and response cycle.
+            """
+        ),
+    ] = None,
 ) -> Any:
     """
     Declare a FastAPI dependency.
@@ -2275,7 +2295,7 @@ async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
         return commons
     ```
     """
-    return params.Depends(dependency=dependency, use_cache=use_cache)
+    return params.Depends(dependency=dependency, use_cache=use_cache, scope=scope)
 
 
 def Security(  # noqa: N802
```

### fastapi/routing.py::request_response.<locals>.app.<locals>.app
```diff
@@ -104,10 +104,11 @@ async def app(scope: Scope, receive: Receive, send: Send) -> None:
         async def app(scope: Scope, receive: Receive, send: Send) -> None:
             # Starts customization
             response_awaited = False
-            async with AsyncExitStack() as stack:
-                scope["fastapi_inner_astack"] = stack
-                # Same as in Starlette
-                response = await f(request)
+            async with AsyncExitStack() as request_stack:
+                scope["fastapi_inner_astack"] = request_stack
+                async with AsyncExitStack() as function_stack:
+                    scope["fastapi_function_astack"] = function_stack
+                    response = await f(request)
                 await response(scope, receive, send)
                 # Continues customization
                 response_awaited = True
```

### fastapi/routing.py::websocket_session.<locals>.app.<locals>.app
```diff
@@ -140,11 +141,11 @@ async def app(scope: Scope, receive: Receive, send: Send) -> None:
         session = WebSocket(scope, receive=receive, send=send)
 
         async def app(scope: Scope, receive: Receive, send: Send) -> None:
-            # Starts customization
-            async with AsyncExitStack() as stack:
-                scope["fastapi_inner_astack"] = stack
-                # Same as in Starlette
-                await func(session)
+            async with AsyncExitStack() as request_stack:
+                scope["fastapi_inner_astack"] = request_stack
+                async with AsyncExitStack() as function_stack:
+                    scope["fastapi_function_astack"] = function_stack
+                    await func(session)
 
         # Same as in Starlette
         await wrap_app_handling_exceptions(app, session)(scope, receive, send)
```

## fastapi_14306
necessary groups: ['fastapi/routing.py::get_request_handler.<locals>.app', 'fastapi/routing.py::get_websocket_app.<locals>.app', 'fastapi/routing.py::serialize_response']

### fastapi/routing.py::get_request_handler.<locals>.app
```diff
@@ -318,6 +350,18 @@ async def app(request: Request) -> Response:
             "fastapi_middleware_astack not found in request scope"
         )
 
+        # Extract endpoint context for error messages
+        endpoint_ctx = (
+            _extract_endpoint_context(dependant.call)
+            if dependant.call
+            else EndpointContext()
+        )
+
+        if dependant.path:
+            # For mounted sub-apps, include the mount path prefix
+            mount_path = request.scope.get("root_path", "").rstrip("/")
+            endpoint_ctx["path"] = f"{request.method} {mount_path}{dependant.path}"
+
         # Read body and auto-close files
         try:
             body: Any = None
@@ -355,6 +399,7 @@ async def app(request: Request) -> Response:
                     }
                 ],
                 body=e.doc,
+                endpoint_ctx=endpoint_ctx,
             )
             raise validation_error from e
         except HTTPException:
@@ -414,14 +459,15 @@ async def app(request: Request) -> Response:
                     exclude_defaults=response_model_exclude_defaults,
                     exclude_none=response_model_exclude_none,
                     is_coroutine=is_coroutine,
+                    endpoint_ctx=endpoint_ctx,
                 )
                 response = actual_response_class(content, **response_args)
                 if not is_body_allowed_for_status_code(response.status_code):
                     response.body = b""
                 response.headers.raw.extend(solved_result.response.headers.raw)
         if errors:
             validation_error = RequestValidationError(
-                _normalize_errors(errors), body=body
```

### fastapi/routing.py::get_websocket_app.<locals>.app
```diff
@@ -438,6 +484,15 @@ def get_websocket_app(
     embed_body_fields: bool = False,
 ) -> Callable[[WebSocket], Coroutine[Any, Any, Any]]:
     async def app(websocket: WebSocket) -> None:
+        endpoint_ctx = (
+            _extract_endpoint_context(dependant.call)
+            if dependant.call
+            else EndpointContext()
+        )
+        if dependant.path:
+            # For mounted sub-apps, include the mount path prefix
+            mount_path = websocket.scope.get("root_path", "").rstrip("/")
+            endpoint_ctx["path"] = f"WS {mount_path}{dependant.path}"
         async_exit_stack = websocket.scope.get("fastapi_inner_astack")
         assert isinstance(async_exit_stack, AsyncExitStack), (
             "fastapi_inner_astack not found in request scope"
@@ -451,7 +506,8 @@ async def app(websocket: WebSocket) -> None:
         )
         if solved_result.errors:
             raise WebSocketRequestValidationError(
-                _normalize_errors(solved_result.errors)
+                _normalize_errors(solved_result.errors),
+                endpoint_ctx=endpoint_ctx,
             )
         assert dependant.call is not None, "dependant.call must be a function"
         await dependant.call(**solved_result.values)```

### fastapi/routing.py::serialize_response
```diff
@@ -223,6 +251,7 @@ async def serialize_response(
     exclude_defaults: bool = False,
     exclude_none: bool = False,
     is_coroutine: bool = True,
+    endpoint_ctx: Optional[EndpointContext] = None,
 ) -> Any:
     if field:
         errors = []
@@ -245,8 +274,11 @@ async def serialize_response(
         elif errors_:
             errors.append(errors_)
         if errors:
+            ctx = endpoint_ctx or EndpointContext()
             raise ResponseValidationError(
-                errors=_normalize_errors(errors), body=response_content
+                errors=_normalize_errors(errors),
+                body=response_content,
+                endpoint_ctx=ctx,
             )
 
         if hasattr(field, "serialize"):
```

## fastapi_14371
necessary groups: ['fastapi/_compat/v2.py::get_schema_from_model_field', 'fastapi/dependencies/utils.py::_extract_form_body', 'fastapi/dependencies/utils.py::_get_multidict_value', 'fastapi/dependencies/utils.py::get_body_field', 'fastapi/dependencies/utils.py::request_body_to_args', 'fastapi/dependencies/utils.py::request_params_to_args', 'fastapi/openapi/utils.py::_get_openapi_operation_parameters', 'fastapi/params.py::Body.__init__']

### fastapi/_compat/v2.py::get_schema_from_model_field
```diff
@@ -243,12 +255,18 @@ def get_schema_from_model_field(
         if (separate_input_output_schemas or _has_computed_fields(field))
         else "validation"
     )
+    field_alias = (
+        (field.validation_alias or field.alias)
+        if field.mode == "validation"
+        else (field.serialization_alias or field.alias)
+    )
+
     # This expects that GenerateJsonSchema was already used to generate the definitions
     json_schema = field_mapping[(field, override_mode or field.mode)]
     if "$ref" not in json_schema:
         # TODO remove when deprecating Pydantic v1
         # Ref: https://github.com/pydantic/pydantic/blob/d61792cc42c80b13b23e3ffa74bc37ec7c77f7d1/pydantic/schema.py#L207
-        json_schema["title"] = field.field_info.title or field.alias.title().replace(
+        json_schema["title"] = field.field_info.title or field_alias.title().replace(
             "_", " "
         )
     return json_schema
```

### fastapi/dependencies/utils.py::_extract_form_body
```diff
@@ -936,8 +934,8 @@ async def process_fn(
                     tg.start_soon(process_fn, sub_value.read)
             value = serialize_sequence_value(field=field, value=results)
         if value is not None:
-            values[field.alias] = value
-    field_aliases = {field.alias for field in body_fields}
+            values[get_validation_alias(field)] = value
+    field_aliases = {get_validation_alias(field) for field in body_fields}
     for key in received_body.keys():
         if key not in field_aliases:
             param_values = received_body.getlist(key)
```

### fastapi/dependencies/utils.py::_get_multidict_value
```diff
@@ -752,7 +752,7 @@ def _validate_value_with_model_field(
 def _get_multidict_value(
     field: ModelField, values: Mapping[str, Any], alias: Union[str, None] = None
 ) -> Any:
-    alias = alias or field.alias
+    alias = alias or get_validation_alias(field)
     if is_sequence_field(field) and isinstance(values, (ImmutableMultiDict, Headers)):
         value = values.getlist(alias)
     else:
```

### fastapi/dependencies/utils.py::get_body_field
```diff
@@ -1062,3 +1060,8 @@ def get_body_field(
         field_info=BodyFieldInfo(**BodyFieldInfo_kwargs),
     )
     return final_field
+
+
+def get_validation_alias(field: ModelField) -> str:
+    va = getattr(field, "validation_alias", None)
+    return va or field.alias
```

### fastapi/dependencies/utils.py::request_body_to_args
```diff
@@ -979,11 +977,11 @@ async def request_body_to_args(
         )
         return {first_field.name: v_}, errors_
     for field in body_fields:
-        loc = ("body", field.alias)
+        loc = ("body", get_validation_alias(field))
         value: Optional[Any] = None
         if body_to_process is not None:
             try:
-                value = body_to_process.get(field.alias)
+                value = body_to_process.get(get_validation_alias(field))
             # If the received body is a list, not a dict
             except AttributeError:
                 errors.append(get_missing_field_error(loc))
```

### fastapi/dependencies/utils.py::request_params_to_args
```diff
@@ -809,15 +809,13 @@ def request_params_to_args(
                 field.field_info, "convert_underscores", default_convert_underscores
             )
             if convert_underscores:
-                alias = (
-                    field.alias
-                    if field.alias != field.name
-                    else field.name.replace("_", "-")
-                )
+                alias = get_validation_alias(field)
+                if alias == field.name:
+                    alias = alias.replace("_", "-")
         value = _get_multidict_value(field, received_params, alias=alias)
         if value is not None:
-            params_to_process[field.alias] = value
-        processed_keys.add(alias or field.alias)
+            params_to_process[get_validation_alias(field)] = value
+        processed_keys.add(alias or get_validation_alias(field))
 
     for key in received_params.keys():
         if key not in processed_keys:
@@ -847,7 +845,7 @@ def request_params_to_args(
         assert isinstance(field_info, (params.Param, temp_pydantic_v1_params.Param)), (
             "Params must be subclasses of Param"
         )
-        loc = (field_info.in_.value, field.alias)
+        loc = (field_info.in_.value, get_validation_alias(field))
         v_, errors_ = _validate_value_with_model_field(
             field=field, value=value, values=values, loc=loc
         )
```

### fastapi/openapi/utils.py::_get_openapi_operation_parameters
```diff
@@ -141,15 +142,15 @@ def _get_openapi_operation_parameters(
                 field_mapping=field_mapping,
                 separate_input_output_schemas=separate_input_output_schemas,
             )
-            name = param.alias
+            name = get_validation_alias(param)
             convert_underscores = getattr(
                 param.field_info,
                 "convert_underscores",
                 default_convert_underscores,
             )
             if (
                 param_type == ParamTypes.header
-                and param.alias == param.name
+                and name == param.name
                 and convert_underscores
             ):
                 name = param.name.replace("_", "-")
```

### fastapi/params.py::Body.__init__
```diff
@@ -571,6 +575,10 @@ def __init__(
         else:
             kwargs["deprecated"] = deprecated
         if PYDANTIC_V2:
+            if serialization_alias in (_Unset, None) and isinstance(alias, str):
+                serialization_alias = alias
+            if validation_alias in (_Unset, None):
+                validation_alias = alias
             kwargs.update(
                 {
                     "annotation": annotation,```

## fastapi_14419
necessary groups: ['fastapi/dependencies/utils.py::get_dependant', 'fastapi/dependencies/utils.py::solve_dependencies']

### fastapi/dependencies/utils.py::get_dependant
```diff
@@ -232,27 +231,27 @@ def get_dependant(
     path: str,
     call: Callable[..., Any],
     name: Optional[str] = None,
-    security_scopes: Optional[List[str]] = None,
+    own_oauth_scopes: Optional[List[str]] = None,
+    parent_oauth_scopes: Optional[List[str]] = None,
     use_cache: bool = True,
     scope: Union[Literal["function", "request"], None] = None,
 ) -> Dependant:
     dependant = Dependant(
         call=call,
         name=name,
         path=path,
-        security_scopes=security_scopes,
         use_cache=use_cache,
         scope=scope,
+        own_oauth_scopes=own_oauth_scopes,
+        parent_oauth_scopes=parent_oauth_scopes,
     )
+    current_scopes = (parent_oauth_scopes or []) + (own_oauth_scopes or [])
     path_param_names = get_path_param_names(path)
     endpoint_signature = get_typed_signature(call)
     signature_params = endpoint_signature.parameters
     if isinstance(call, SecurityBase):
-        use_scopes: List[str] = []
-        if isinstance(call, (OAuth2, OpenIdConnect)):
-            use_scopes = security_scopes or use_scopes
         security_requirement = SecurityRequirement(
-            security_scheme=call, scopes=use_scopes
+            security_scheme=call, scopes=current_scopes
         )
         dependant.security_requirements.append(security_requirement)
     for param_name, param in signature_params.items():
@@ -275,17 +274,16 @@ def get_dependant(
                     f'The dependency "{dependant.call.__name__}" has a scope of '
                     '"request", it cannot depend on dependencies with scope "function".'
                 )
-            use_security_scopes = security_scopes or []
+            sub_own_oauth_scopes: List[str] = []
```

### fastapi/dependencies/utils.py::solve_dependencies
```diff
@@ -611,7 +609,7 @@ async def solve_dependencies(
                 path=use_path,
                 call=call,
                 name=sub_dependant.name,
-                security_scopes=sub_dependant.security_scopes,
+                parent_oauth_scopes=sub_dependant.oauth_scopes,
                 scope=sub_dependant.scope,
             )
 
@@ -693,7 +691,7 @@ async def solve_dependencies(
         values[dependant.response_param_name] = response
     if dependant.security_scopes_param_name:
         values[dependant.security_scopes_param_name] = SecurityScopes(
-            scopes=dependant.security_scopes
+            scopes=dependant.oauth_scopes
         )
     return SolvedDependency(
         values=values,```

## fastapi_14458
necessary groups: ['fastapi/dependencies/models.py::Dependant.is_async_gen_callable', 'fastapi/dependencies/models.py::Dependant.is_coroutine_callable', 'fastapi/dependencies/models.py::Dependant.is_gen_callable']

### fastapi/dependencies/models.py::Dependant.is_async_gen_callable
```diff
@@ -134,6 +136,8 @@ def is_async_gen_callable(self) -> bool:
             _impartial(self.call)
         ) or inspect.isasyncgenfunction(_unwrapped_call(self.call)):
             return True
+        if inspect.isclass(_unwrapped_call(self.call)):
+            return False
         dunder_call = getattr(_impartial(self.call), "__call__", None)  # noqa: B004
         if dunder_call is None:
             return False  # pragma: no cover
```

### fastapi/dependencies/models.py::Dependant.is_coroutine_callable
```diff
@@ -162,6 +166,8 @@ def is_coroutine_callable(self) -> bool:
             _unwrapped_call(self.call)
         ):
             return True
+        if inspect.isclass(_unwrapped_call(self.call)):
+            return False
         dunder_call = getattr(_impartial(self.call), "__call__", None)  # noqa: B004
         if dunder_call is None:
             return False  # pragma: no cover
@@ -176,7 +182,6 @@ def is_coroutine_callable(self) -> bool:
             _impartial(dunder_unwrapped_call)
         ) or iscoroutinefunction(_unwrapped_call(dunder_unwrapped_call)):
             return True
-        # if inspect.isclass(self.call): False, covered by default return
         return False
 
     @cached_property```

### fastapi/dependencies/models.py::Dependant.is_gen_callable
```diff
@@ -110,6 +110,8 @@ def is_gen_callable(self) -> bool:
             _impartial(self.call)
         ) or inspect.isgeneratorfunction(_unwrapped_call(self.call)):
             return True
+        if inspect.isclass(_unwrapped_call(self.call)):
+            return False
         dunder_call = getattr(_impartial(self.call), "__call__", None)  # noqa: B004
         if dunder_call is None:
             return False  # pragma: no cover
```

## fastapi_14459
necessary groups: ['fastapi/dependencies/utils.py::get_dependant', 'fastapi/dependencies/utils.py::get_flat_dependant', 'fastapi/openapi/utils.py::get_openapi_security_definitions']

### fastapi/dependencies/utils.py::get_dependant
```diff
@@ -258,11 +276,6 @@ def get_dependant(
     path_param_names = get_path_param_names(path)
     endpoint_signature = get_typed_signature(call)
     signature_params = endpoint_signature.parameters
-    if isinstance(call, SecurityBase):
-        security_requirement = SecurityRequirement(
-            security_scheme=call, scopes=current_scopes
-        )
-        dependant.security_requirements.append(security_requirement)
     for param_name, param in signature_params.items():
         is_path_param = param_name in path_param_names
         param_details = analyze_param(
```

### fastapi/dependencies/utils.py::get_flat_dependant
```diff
@@ -142,33 +141,52 @@ def get_flat_dependant(
     *,
     skip_repeats: bool = False,
     visited: Optional[List[DependencyCacheKey]] = None,
+    parent_oauth_scopes: Optional[List[str]] = None,
 ) -> Dependant:
     if visited is None:
         visited = []
     visited.append(dependant.cache_key)
+    use_parent_oauth_scopes = (parent_oauth_scopes or []) + (
+        dependant.oauth_scopes or []
+    )
 
     flat_dependant = Dependant(
         path_params=dependant.path_params.copy(),
         query_params=dependant.query_params.copy(),
         header_params=dependant.header_params.copy(),
         cookie_params=dependant.cookie_params.copy(),
         body_params=dependant.body_params.copy(),
-        security_requirements=dependant.security_requirements.copy(),
+        name=dependant.name,
+        call=dependant.call,
+        request_param_name=dependant.request_param_name,
+        websocket_param_name=dependant.websocket_param_name,
+        http_connection_param_name=dependant.http_connection_param_name,
+        response_param_name=dependant.response_param_name,
+        background_tasks_param_name=dependant.background_tasks_param_name,
+        security_scopes_param_name=dependant.security_scopes_param_name,
+        own_oauth_scopes=dependant.own_oauth_scopes,
+        parent_oauth_scopes=use_parent_oauth_scopes,
         use_cache=dependant.use_cache,
         path=dependant.path,
+        scope=dependant.scope,
     )
     for sub_dependant in dependant.dependencies:
         if skip_repeats and sub_dependant.cache_key in visited:
             continue
         flat_sub = get_flat_dependant(
-            sub_dependant, skip_repeats=skip_repeats, visited=visited
+            sub_dependant,
```

### fastapi/openapi/utils.py::get_openapi_security_definitions
```diff
@@ -81,18 +81,18 @@ def get_openapi_security_definitions(
     security_definitions = {}
     # Use a dict to merge scopes for same security scheme
     operation_security_dict: Dict[str, List[str]] = {}
-    for security_requirement in flat_dependant.security_requirements:
+    for security_dependency in flat_dependant._security_dependencies:
         security_definition = jsonable_encoder(
-            security_requirement.security_scheme.model,
+            security_dependency._security_scheme.model,
             by_alias=True,
             exclude_none=True,
         )
-        security_name = security_requirement.security_scheme.scheme_name
+        security_name = security_dependency._security_scheme.scheme_name
         security_definitions[security_name] = security_definition
         # Merge scopes for the same security scheme
         if security_name not in operation_security_dict:
             operation_security_dict[security_name] = []
-        for scope in security_requirement.scopes or []:
+        for scope in security_dependency.oauth_scopes or []:
             if scope not in operation_security_dict[security_name]:
                 operation_security_dict[security_name].append(scope)
     operation_security = [```

## fastapi_14482
necessary groups: ['fastapi/_compat/v2.py::ModelField.__post_init__', 'fastapi/_compat/v2.py::get_model_fields']

### fastapi/_compat/v2.py::ModelField.__post_init__
```diff
@@ -94,8 +95,14 @@ def __post_init__(self) -> None:
                 warnings.simplefilter(
                     "ignore", category=UnsupportedFieldAttributeWarning
                 )
+            annotated_args = (
+                self.field_info.annotation,
+                *self.field_info.metadata,
+                self.field_info,
+            )
             self._type_adapter: TypeAdapter[Any] = TypeAdapter(
-                Annotated[self.field_info.annotation, self.field_info]
+                Annotated[annotated_args],
+                config=self.config,
             )
 
     def get_default(self) -> Any:
```

### fastapi/_compat/v2.py::get_model_fields
```diff
@@ -412,10 +419,21 @@ def create_body_model(
 
 
 def get_model_fields(model: Type[BaseModel]) -> List[ModelField]:
-    return [
-        ModelField(field_info=field_info, name=name)
-        for name, field_info in model.model_fields.items()
-    ]
+    model_fields: List[ModelField] = []
+    for name, field_info in model.model_fields.items():
+        type_ = field_info.annotation
+        if lenient_issubclass(type_, (BaseModel, dict)) or is_dataclass(type_):
+            model_config = None
+        else:
+            model_config = model.model_config
+        model_fields.append(
+            ModelField(
+                field_info=field_info,
+                name=name,
+                config=model_config,
+            )
+        )
+    return model_fields
 
 
 # Duplicate of several schema functions from Pydantic v1 to make them compatible with```

## fastapi_14583
necessary groups: ['fastapi/dependencies/utils.py::get_dependant', 'fastapi/routing.py::APIRoute.__init__']

### fastapi/dependencies/utils.py::get_dependant
```diff
@@ -322,6 +323,13 @@ def get_dependant(
             )
             continue
         assert param_details.field is not None
+        if isinstance(param_details.field, may_v1.ModelField):
+            warnings.warn(
+                "pydantic.v1 is deprecated and will soon stop being supported by FastAPI."
+                f" Please update the param {param_name}: {param_details.type_annotation!r}.",
+                category=DeprecationWarning,
+                stacklevel=5,
+            )
         if isinstance(
             param_details.field.field_info, (params.Body, temp_pydantic_v1_params.Body)
         ):
```

### fastapi/routing.py::APIRoute.__init__
```diff
@@ -634,6 +636,13 @@ def __init__(
                 f"Status code {status_code} must not have a response body"
             )
             response_name = "Response_" + self.unique_id
+            if annotation_is_pydantic_v1(self.response_model):
+                warnings.warn(
+                    "pydantic.v1 is deprecated and will soon stop being supported by FastAPI."
+                    f" Please update the response model {self.response_model!r}.",
+                    category=DeprecationWarning,
+                    stacklevel=4,
+                )
             self.response_field = create_model_field(
                 name=response_name,
                 type_=self.response_model,
@@ -667,6 +676,13 @@ def __init__(
                     f"Status code {additional_status_code} must not have a response body"
                 )
                 response_name = f"Response_{additional_status_code}_{self.unique_id}"
+                if annotation_is_pydantic_v1(model):
+                    warnings.warn(
+                        "pydantic.v1 is deprecated and will soon stop being supported by FastAPI."
+                        f" In responses={{}}, please update {model}.",
+                        category=DeprecationWarning,
+                        stacklevel=4,
+                    )
                 response_field = create_model_field(
                     name=response_name, type_=model, mode="serialization"
                 )```

## fastapi_14605
necessary groups: ['fastapi/dependencies/utils.py::get_dependant', 'fastapi/exceptions.py::ResponseValidationError.__init__', 'fastapi/params.py::Body.__init__', 'fastapi/params.py::Param.__init__', 'fastapi/routing.py::APIRoute.__init__', 'fastapi/temp_pydantic_v1_params.py::Body.__init__', 'fastapi/temp_pydantic_v1_params.py::Param.__init__']

### fastapi/dependencies/utils.py::get_dependant
```diff
@@ -327,7 +327,7 @@ def get_dependant(
             warnings.warn(
                 "pydantic.v1 is deprecated and will soon stop being supported by FastAPI."
                 f" Please update the param {param_name}: {param_details.type_annotation!r}.",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=5,
             )
         if isinstance(
```

### fastapi/exceptions.py::ResponseValidationError.__init__
```diff
@@ -231,3 +231,10 @@ def __init__(
     ) -> None:
         super().__init__(errors, endpoint_ctx=endpoint_ctx)
         self.body = body
+
+
+class FastAPIDeprecationWarning(UserWarning):
+    """
+    A custom deprecation warning as DeprecationWarning is ignored
+    Ref: https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries
+    """
```

### fastapi/params.py::Body.__init__
```diff
@@ -530,7 +531,7 @@ def __init__(
         if example is not _Unset:
             warnings.warn(
                 "`example` has been deprecated, please use `examples` instead",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=4,
             )
         self.example = example
@@ -560,7 +561,7 @@ def __init__(
         if regex is not None:
             warnings.warn(
                 "`regex` has been deprecated, please use `pattern` instead",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=4,
             )
         current_json_schema_extra = json_schema_extra or extra
```

### fastapi/params.py::Param.__init__
```diff
@@ -75,7 +76,7 @@ def __init__(
         if example is not _Unset:
             warnings.warn(
                 "`example` has been deprecated, please use `examples` instead",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=4,
             )
         self.example = example
@@ -105,7 +106,7 @@ def __init__(
         if regex is not None:
             warnings.warn(
                 "`regex` has been deprecated, please use `pattern` instead",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=4,
             )
         current_json_schema_extra = json_schema_extra or extra
```

### fastapi/routing.py::APIRoute.__init__
```diff
@@ -640,7 +641,7 @@ def __init__(
                 warnings.warn(
                     "pydantic.v1 is deprecated and will soon stop being supported by FastAPI."
                     f" Please update the response model {self.response_model!r}.",
-                    category=DeprecationWarning,
+                    category=FastAPIDeprecationWarning,
                     stacklevel=4,
                 )
             self.response_field = create_model_field(
@@ -680,7 +681,7 @@ def __init__(
                     warnings.warn(
                         "pydantic.v1 is deprecated and will soon stop being supported by FastAPI."
                         f" In responses={{}}, please update {model}.",
-                        category=DeprecationWarning,
+                        category=FastAPIDeprecationWarning,
                         stacklevel=4,
                     )
                 response_field = create_model_field(
```

### fastapi/temp_pydantic_v1_params.py::Body.__init__
```diff
@@ -503,7 +504,7 @@ def __init__(
         if example is not _Unset:
             warnings.warn(
                 "`example` has been deprecated, please use `examples` instead",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=4,
             )
         self.example = example
@@ -533,7 +534,7 @@ def __init__(
         if regex is not None:
             warnings.warn(
                 "`regex` has been deprecated, please use `pattern` instead",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=4,
             )
         current_json_schema_extra = json_schema_extra or extra
```

### fastapi/temp_pydantic_v1_params.py::Param.__init__
```diff
@@ -63,7 +64,7 @@ def __init__(
         if example is not _Unset:
             warnings.warn(
                 "`example` has been deprecated, please use `examples` instead",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=4,
             )
         self.example = example
@@ -93,7 +94,7 @@ def __init__(
         if regex is not None:
             warnings.warn(
                 "`regex` has been deprecated, please use `pattern` instead",
-                category=DeprecationWarning,
+                category=FastAPIDeprecationWarning,
                 stacklevel=4,
             )
         current_json_schema_extra = json_schema_extra or extra
```

## fastapi_14609
necessary groups: ['fastapi/_compat/shared.py::_annotation_is_complex', 'fastapi/_compat/v2.py::get_long_model_name', 'fastapi/datastructures.py::UploadFile.__get_pydantic_core_schema__ | fastapi/datastructures.py::UploadFile.__get_pydantic_json_schema__', 'fastapi/dependencies/utils.py::_get_flat_fields_from_params', 'fastapi/dependencies/utils.py::_get_multidict_value', 'fastapi/dependencies/utils.py::_should_embed_body_fields', 'fastapi/dependencies/utils.py::_validate_value_with_model_field', 'fastapi/dependencies/utils.py::analyze_param', 'fastapi/dependencies/utils.py::get_dependant', 'fastapi/dependencies/utils.py::request_body_to_args', 'fastapi/dependencies/utils.py::request_params_to_args', 'fastapi/encoders.py::jsonable_encoder', 'fastapi/openapi/models.py::EmailStr.__get_pydantic_core_schema__ | fastapi/openapi/models.py::EmailStr.__get_pydantic_json_schema__', 'fastapi/openapi/utils.py::_get_openapi_operation_parameters', 'fastapi/openapi/utils.py::get_openapi_operation_request_body', 'fastapi/openapi/utils.py::get_openapi_path', 'fastapi/routing.py::get_request_handler', 'fastapi/utils.py::create_model_field']

### fastapi/_compat/shared.py::_annotation_is_complex
```diff
@@ -81,9 +81,7 @@ def value_is_sequence(value: Any) -> bool:
 
 def _annotation_is_complex(annotation: Union[type[Any], None]) -> bool:
     return (
-        lenient_issubclass(
-            annotation, (BaseModel, may_v1.BaseModel, Mapping, UploadFile)
-        )
+        lenient_issubclass(annotation, (BaseModel, Mapping, UploadFile))
         or _annotation_is_sequence(annotation)
         or is_dataclass(annotation)
     )
```

### fastapi/_compat/v2.py::get_long_model_name
```diff
@@ -567,5 +558,11 @@ def get_flat_models_from_fields(
     return known_models
 
 
-def get_long_model_name(model: TypeModelOrEnum) -> str:
-    return f"{model.__module__}__{model.__qualname__}".replace(".", "__")
+def _regenerate_error_with_loc(
+    *, errors: Sequence[Any], loc_prefix: tuple[Union[str, int], ...]
+) -> list[dict[str, Any]]:
+    updated_loc_errors: list[Any] = [
+        {**err, "loc": loc_prefix + err.get("loc", ())} for err in errors
+    ]
+
+    return updated_loc_errors
```

### fastapi/datastructures.py::UploadFile.__get_pydantic_core_schema__ | fastapi/datastructures.py::UploadFile.__get_pydantic_json_schema__
```diff
@@ -142,14 +139,14 @@ def _validate(cls, __input_value: Any, _: Any) -> "UploadFile":
 
     @classmethod
     def __get_pydantic_json_schema__(
-        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
-    ) -> JsonSchemaValue:
+        cls, core_schema: Mapping[str, Any], handler: GetJsonSchemaHandler
+    ) -> dict[str, Any]:
         return {"type": "string", "format": "binary"}
 
     @classmethod
     def __get_pydantic_core_schema__(
-        cls, source: type[Any], handler: Callable[[Any], CoreSchema]
-    ) -> CoreSchema:
+        cls, source: type[Any], handler: Callable[[Any], Mapping[str, Any]]
+    ) -> Mapping[str, Any]:
         from ._compat.v2 import with_info_plain_validator_function
 
         return with_info_plain_validator_function(cls._validate)
```

### fastapi/dependencies/utils.py::_get_flat_fields_from_params
```diff
@@ -189,7 +182,7 @@ def _get_flat_fields_from_params(fields: list[ModelField]) -> list[ModelField]:
     if not fields:
         return fields
     first_field = fields[0]
-    if len(fields) == 1 and _is_model_class(first_field.type_):
+    if len(fields) == 1 and lenient_issubclass(first_field.type_, BaseModel):
         fields_to_extract = get_cached_model_fields(first_field.type_)
         return fields_to_extract
     return fields
```

### fastapi/dependencies/utils.py::_get_multidict_value
```diff
@@ -762,7 +727,7 @@ def _get_multidict_value(
     if (
         value is None
         or (
-            isinstance(field.field_info, (params.Form, temp_pydantic_v1_params.Form))
+            isinstance(field.field_info, params.Form)
             and isinstance(value, str)  # For type checks
             and value == ""
         )
```

### fastapi/dependencies/utils.py::_should_embed_body_fields
```diff
@@ -893,8 +858,8 @@ def _should_embed_body_fields(fields: list[ModelField]) -> bool:
     # If it's a Form (or File) field, it has to be a BaseModel (or a union of BaseModels) to be top level
     # otherwise it has to be embedded, so that the key value pair can be extracted
     if (
-        isinstance(first_field.field_info, (params.Form, temp_pydantic_v1_params.Form))
-        and not _is_model_class(first_field.type_)
+        isinstance(first_field.field_info, params.Form)
+        and not lenient_issubclass(first_field.type_, BaseModel)
         and not is_union_of_base_models(first_field.type_)
     ):
         return True
```

### fastapi/dependencies/utils.py::_validate_value_with_model_field
```diff
@@ -742,10 +709,8 @@ def _validate_value_with_model_field(
         else:
             return deepcopy(field.default), []
     v_, errors_ = field.validate(value, values, loc=loc)
-    if _is_error_wrapper(errors_):  # type: ignore[arg-type]
-        return None, [errors_]
-    elif isinstance(errors_, list):
-        new_errors = may_v1._regenerate_error_with_loc(errors=errors_, loc_prefix=())
+    if isinstance(errors_, list):
+        new_errors = _regenerate_error_with_loc(errors=errors_, loc_prefix=())
         return None, new_errors
     else:
         return v_, []
```

### fastapi/dependencies/utils.py::analyze_param
```diff
@@ -391,7 +375,7 @@ def analyze_param(
         fastapi_annotations = [
             arg
             for arg in annotated_args[1:]
-            if isinstance(arg, (FieldInfo, may_v1.FieldInfo, params.Depends))
+            if isinstance(arg, (FieldInfo, params.Depends))
         ]
         fastapi_specific_annotations = [
             arg
@@ -400,30 +384,27 @@ def analyze_param(
                 arg,
                 (
                     params.Param,
-                    temp_pydantic_v1_params.Param,
                     params.Body,
-                    temp_pydantic_v1_params.Body,
                     params.Depends,
                 ),
             )
         ]
         if fastapi_specific_annotations:
-            fastapi_annotation: Union[
-                FieldInfo, may_v1.FieldInfo, params.Depends, None
-            ] = fastapi_specific_annotations[-1]
+            fastapi_annotation: Union[FieldInfo, params.Depends, None] = (
+                fastapi_specific_annotations[-1]
+            )
         else:
             fastapi_annotation = None
         # Set default for Annotated FieldInfo
-        if isinstance(fastapi_annotation, (FieldInfo, may_v1.FieldInfo)):
+        if isinstance(fastapi_annotation, FieldInfo):
             # Copy `field_info` because we mutate `field_info.default` below.
             field_info = copy_field_info(
                 field_info=fastapi_annotation,  # type: ignore[arg-type]
                 annotation=use_annotation,
             )
-            assert field_info.default in {
-                Undefined,
-                may_v1.Undefined,
```

### fastapi/dependencies/utils.py::get_dependant
```diff
@@ -323,16 +316,7 @@ def get_dependant(
             )
             continue
         assert param_details.field is not None
-        if isinstance(param_details.field, may_v1.ModelField):
-            warnings.warn(
-                "pydantic.v1 is deprecated and will soon stop being supported by FastAPI."
-                f" Please update the param {param_name}: {param_details.type_annotation!r}.",
-                category=FastAPIDeprecationWarning,
-                stacklevel=5,
-            )
-        if isinstance(
-            param_details.field.field_info, (params.Body, temp_pydantic_v1_params.Body)
-        ):
+        if isinstance(param_details.field.field_info, params.Body):
             dependant.body_params.append(param_details.field)
         else:
             add_param_to_fields(field=param_details.field, dependant=dependant)
```

### fastapi/dependencies/utils.py::request_body_to_args
```diff
@@ -964,7 +929,7 @@ async def request_body_to_args(
 
     if (
         single_not_embedded_field
-        and _is_model_class(first_field.type_)
+        and lenient_issubclass(first_field.type_, BaseModel)
         and isinstance(received_body, FormData)
     ):
         fields_to_extract = get_cached_model_fields(first_field.type_)
```

### fastapi/dependencies/utils.py::request_params_to_args
```diff
@@ -832,7 +797,7 @@ def request_params_to_args(
 
     if single_not_embedded_field:
         field_info = first_field.field_info
-        assert isinstance(field_info, (params.Param, temp_pydantic_v1_params.Param)), (
+        assert isinstance(field_info, params.Param), (
             "Params must be subclasses of Param"
         )
         loc: tuple[str, ...] = (field_info.in_.value,)
@@ -844,7 +809,7 @@ def request_params_to_args(
     for field in fields:
         value = _get_multidict_value(field, received_params)
         field_info = field.field_info
-        assert isinstance(field_info, (params.Param, temp_pydantic_v1_params.Param)), (
+        assert isinstance(field_info, params.Param), (
             "Params must be subclasses of Param"
         )
         loc = (field_info.in_.value, get_validation_alias(field))
```

### fastapi/encoders.py::jsonable_encoder
```diff
@@ -224,15 +222,8 @@ def jsonable_encoder(
         include = set(include)
     if exclude is not None and not isinstance(exclude, (set, dict)):
         exclude = set(exclude)
-    if isinstance(obj, (BaseModel, may_v1.BaseModel)):
-        # TODO: remove when deprecating Pydantic v1
-        encoders: dict[Any, Any] = {}
-        if isinstance(obj, may_v1.BaseModel):
-            encoders = getattr(obj.__config__, "json_encoders", {})
-            if custom_encoder:
-                encoders = {**encoders, **custom_encoder}
-        obj_dict = _model_dump(
-            obj,  # type: ignore[arg-type]
+    if isinstance(obj, BaseModel):
+        obj_dict = obj.model_dump(
             mode="json",
             include=include,
             exclude=exclude,
@@ -241,14 +232,10 @@ def jsonable_encoder(
             exclude_none=exclude_none,
             exclude_defaults=exclude_defaults,
         )
-        if "__root__" in obj_dict:
-            obj_dict = obj_dict["__root__"]
         return jsonable_encoder(
             obj_dict,
             exclude_none=exclude_none,
             exclude_defaults=exclude_defaults,
-            # TODO: remove when deprecating Pydantic v1
-            custom_encoder=encoders,
             sqlalchemy_safe=sqlalchemy_safe,
         )
     if dataclasses.is_dataclass(obj):
@@ -271,7 +258,7 @@ def jsonable_encoder(
         return str(obj)
     if isinstance(obj, (str, int, float, type(None))):
         return obj
-    if _is_undefined(obj):
+    if isinstance(obj, PydanticUndefinedType):
         return None
```

### fastapi/openapi/models.py::EmailStr.__get_pydantic_core_schema__ | fastapi/openapi/models.py::EmailStr.__get_pydantic_json_schema__
```diff
@@ -43,14 +43,14 @@ def _validate(cls, __input_value: Any, _: Any) -> str:
 
         @classmethod
         def __get_pydantic_json_schema__(
-            cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
-        ) -> JsonSchemaValue:
+            cls, core_schema: Mapping[str, Any], handler: GetJsonSchemaHandler
+        ) -> dict[str, Any]:
             return {"type": "string", "format": "email"}
 
         @classmethod
         def __get_pydantic_core_schema__(
-            cls, source: type[Any], handler: Callable[[Any], CoreSchema]
-        ) -> CoreSchema:
+            cls, source: type[Any], handler: Callable[[Any], Mapping[str, Any]]
+        ) -> Mapping[str, Any]:
             return with_info_plain_validator_function(cls._validate)
 
 
```

### fastapi/openapi/utils.py::_get_openapi_operation_parameters
```diff
@@ -109,7 +106,7 @@ def _get_openapi_operation_parameters(
     dependant: Dependant,
     model_name_map: ModelNameMap,
     field_mapping: dict[
-        tuple[ModelField, Literal["validation", "serialization"]], JsonSchemaValue
+        tuple[ModelField, Literal["validation", "serialization"]], dict[str, Any]
     ],
     separate_input_output_schemas: bool = True,
 ) -> list[dict[str, Any]]:
```

### fastapi/openapi/utils.py::get_openapi_operation_request_body
```diff
@@ -182,13 +179,13 @@ def get_openapi_operation_request_body(
     body_field: Optional[ModelField],
     model_name_map: ModelNameMap,
     field_mapping: dict[
-        tuple[ModelField, Literal["validation", "serialization"]], JsonSchemaValue
+        tuple[ModelField, Literal["validation", "serialization"]], dict[str, Any]
     ],
     separate_input_output_schemas: bool = True,
 ) -> Optional[dict[str, Any]]:
     if not body_field:
         return None
-    assert _is_model_field(body_field)
+    assert isinstance(body_field, ModelField)
     body_schema = get_schema_from_model_field(
         field=body_field,
         model_name_map=model_name_map,
```

### fastapi/openapi/utils.py::get_openapi_path
```diff
@@ -265,7 +262,7 @@ def get_openapi_path(
     operation_ids: set[str],
     model_name_map: ModelNameMap,
     field_mapping: dict[
-        tuple[ModelField, Literal["validation", "serialization"]], JsonSchemaValue
+        tuple[ModelField, Literal["validation", "serialization"]], dict[str, Any]
     ],
     separate_input_output_schemas: bool = True,
 ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
```

### fastapi/routing.py::get_request_handler
```diff
@@ -332,9 +262,7 @@ def get_request_handler(
 ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
     assert dependant.call is not None, "dependant.call must be a function"
     is_coroutine = dependant.is_coroutine_callable
-    is_body_form = body_field and isinstance(
-        body_field.field_info, (params.Form, temp_pydantic_v1_params.Form)
-    )
+    is_body_form = body_field and isinstance(body_field.field_info, params.Form)
     if isinstance(response_class, DefaultPlaceholder):
         actual_response_class: type[Response] = response_class.value
     else:
```

### fastapi/utils.py::create_model_field
```diff
@@ -83,52 +80,18 @@ def create_model_field(
     mode: Literal["validation", "serialization"] = "validation",
     version: Literal["1", "auto"] = "auto",
 ) -> ModelField:
-    class_validators = class_validators or {}
-
-    v1_model_config = may_v1.BaseConfig
-    v1_field_info = field_info or may_v1.FieldInfo()
-    v1_kwargs = {
-        "name": name,
-        "field_info": v1_field_info,
-        "type_": type_,
-        "class_validators": class_validators,
-        "default": default,
-        "required": required,
-        "model_config": v1_model_config,
-        "alias": alias,
-    }
-
-    if (
-        annotation_is_pydantic_v1(type_)
-        or isinstance(field_info, may_v1.FieldInfo)
-        or version == "1"
-    ):
-        from fastapi._compat import v1
-
-        try:
-            return v1.ModelField(**v1_kwargs)  # type: ignore[return-value]
-        except RuntimeError:
-            raise fastapi.exceptions.FastAPIError(
-                _invalid_args_message.format(type_=type_)
-            ) from None
-    else:
-        field_info = field_info or FieldInfo(
-            annotation=type_, default=default, alias=alias
+    if annotation_is_pydantic_v1(type_):
+        raise PydanticV1NotSupportedError(
+            "pydantic.v1 models are no longer supported by FastAPI."
+            f" Please update the response model {type_!r}."
         )
```

## fastapi_14978
necessary groups: ['fastapi/applications.py::FastAPI.__init__', 'fastapi/routing.py::APIRoute.__init__', 'fastapi/routing.py::APIRoute.get_route_handler', 'fastapi/routing.py::APIRouter.__init__', 'fastapi/routing.py::APIRouter.add_api_route', 'fastapi/routing.py::APIRouter.include_router', 'fastapi/routing.py::get_request_handler', 'fastapi/routing.py::get_request_handler.<locals>.app']

### fastapi/applications.py::FastAPI.__init__
```diff
@@ -840,6 +840,29 @@ class Item(BaseModel):
                 """
             ),
         ] = None,
+        strict_content_type: Annotated[
+            bool,
+            Doc(
+                """
+                Enable strict checking for request Content-Type headers.
+
+                When `True` (the default), requests with a body that do not include
+                a `Content-Type` header will **not** be parsed as JSON.
+
+                This prevents potential cross-site request forgery (CSRF) attacks
+                that exploit the browser's ability to send requests without a
+                Content-Type header, bypassing CORS preflight checks. In particular
+                applicable for apps that need to be run locally (in localhost).
+
+                When `False`, requests without a `Content-Type` header will have
+                their body parsed as JSON, which maintains compatibility with
+                certain clients that don't send `Content-Type` headers.
+
+                Read more about it in the
+                [FastAPI docs for Strict Content-Type](https://fastapi.tiangolo.com/advanced/strict-content-type/).
+                """
+            ),
+        ] = True,
         **extra: Annotated[
             Any,
             Doc(
@@ -974,6 +997,7 @@ class Item(BaseModel):
             include_in_schema=include_in_schema,
             responses=responses,
             generate_unique_id_function=generate_unique_id_function,
+            strict_content_type=strict_content_type,
         )
         self.exception_handlers: dict[
             Any, Callable[[Request, Any], Response | Awaitable[Response]]
```

### fastapi/routing.py::APIRoute.__init__
```diff
@@ -599,6 +605,7 @@ def __init__(
         openapi_extra: dict[str, Any] | None = None,
         generate_unique_id_function: Callable[["APIRoute"], str]
         | DefaultPlaceholder = Default(generate_unique_id),
+        strict_content_type: bool | DefaultPlaceholder = Default(True),
     ) -> None:
         self.path = path
         self.endpoint = endpoint
@@ -625,6 +632,7 @@ def __init__(
         self.callbacks = callbacks
         self.openapi_extra = openapi_extra
         self.generate_unique_id_function = generate_unique_id_function
+        self.strict_content_type = strict_content_type
         self.tags = tags or []
         self.responses = responses or {}
         self.name = get_name(endpoint) if name is None else name
```

### fastapi/routing.py::APIRoute.get_route_handler
```diff
@@ -713,6 +721,7 @@ def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]
             response_model_exclude_none=self.response_model_exclude_none,
             dependency_overrides_provider=self.dependency_overrides_provider,
             embed_body_fields=self._embed_body_fields,
+            strict_content_type=self.strict_content_type,
         )
 
     def matches(self, scope: Scope) -> tuple[Match, Scope]:
```

### fastapi/routing.py::APIRouter.__init__
```diff
@@ -963,6 +972,29 @@ def __init__(
                 """
             ),
         ] = Default(generate_unique_id),
+        strict_content_type: Annotated[
+            bool,
+            Doc(
+                """
+                Enable strict checking for request Content-Type headers.
+
+                When `True` (the default), requests with a body that do not include
+                a `Content-Type` header will **not** be parsed as JSON.
+
+                This prevents potential cross-site request forgery (CSRF) attacks
+                that exploit the browser's ability to send requests without a
+                Content-Type header, bypassing CORS preflight checks. In particular
+                applicable for apps that need to be run locally (in localhost).
+
+                When `False`, requests without a `Content-Type` header will have
+                their body parsed as JSON, which maintains compatibility with
+                certain clients that don't send `Content-Type` headers.
+
+                Read more about it in the
+                [FastAPI docs for Strict Content-Type](https://fastapi.tiangolo.com/advanced/strict-content-type/).
+                """
+            ),
+        ] = Default(True),
     ) -> None:
         # Determine the lifespan context to use
         if lifespan is None:
@@ -1009,6 +1041,7 @@ def __init__(
         self.route_class = route_class
         self.default_response_class = default_response_class
         self.generate_unique_id_function = generate_unique_id_function
+        self.strict_content_type = strict_content_type
 
     def route(
         self,
```

### fastapi/routing.py::APIRouter.add_api_route
```diff
@@ -1059,6 +1092,7 @@ def add_api_route(
         openapi_extra: dict[str, Any] | None = None,
         generate_unique_id_function: Callable[[APIRoute], str]
         | DefaultPlaceholder = Default(generate_unique_id),
+        strict_content_type: bool | DefaultPlaceholder = Default(True),
     ) -> None:
         route_class = route_class_override or self.route_class
         responses = responses or {}
@@ -1105,6 +1139,9 @@ def add_api_route(
             callbacks=current_callbacks,
             openapi_extra=openapi_extra,
             generate_unique_id_function=current_generate_unique_id,
+            strict_content_type=get_value_or_default(
+                strict_content_type, self.strict_content_type
+            ),
         )
         self.routes.append(route)
 
```

### fastapi/routing.py::APIRouter.include_router
```diff
@@ -1480,6 +1517,11 @@ def read_users():
                     callbacks=current_callbacks,
                     openapi_extra=route.openapi_extra,
                     generate_unique_id_function=current_generate_unique_id,
+                    strict_content_type=get_value_or_default(
+                        route.strict_content_type,
+                        router.strict_content_type,
+                        self.strict_content_type,
+                    ),
                 )
             elif isinstance(route, routing.Route):
                 methods = list(route.methods or [])```

### fastapi/routing.py::get_request_handler
```diff
@@ -329,6 +329,7 @@ def get_request_handler(
     response_model_exclude_none: bool = False,
     dependency_overrides_provider: Any | None = None,
     embed_body_fields: bool = False,
+    strict_content_type: bool | DefaultPlaceholder = Default(True),
 ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
     assert dependant.call is not None, "dependant.call must be a function"
     is_coroutine = dependant.is_coroutine_callable
@@ -337,6 +338,10 @@ def get_request_handler(
         actual_response_class: type[Response] = response_class.value
     else:
         actual_response_class = response_class
+    if isinstance(strict_content_type, DefaultPlaceholder):
+        actual_strict_content_type: bool = strict_content_type.value
+    else:
+        actual_strict_content_type = strict_content_type
 
     async def app(request: Request) -> Response:
         response: Response | None = None
```

### fastapi/routing.py::get_request_handler.<locals>.app
```diff
@@ -370,7 +375,8 @@ async def app(request: Request) -> Response:
                         json_body: Any = Undefined
                         content_type_value = request.headers.get("content-type")
                         if not content_type_value:
-                            json_body = await request.json()
+                            if not actual_strict_content_type:
+                                json_body = await request.json()
                         else:
                             message = email.message.Message()
                             message["content-type"] = content_type_value
```

## fastapi_15030
necessary groups: ['fastapi/openapi/utils.py::get_openapi_path', 'fastapi/routing.py::APIRoute.__init__', 'fastapi/routing.py::get_request_handler', 'fastapi/routing.py::get_request_handler.<locals>.app | fastapi/routing.py::get_request_handler.<locals>.app.<locals>._serialize_item']

### fastapi/openapi/utils.py::get_openapi_path
```diff
@@ -372,6 +373,26 @@ def get_openapi_path(
                     operation.setdefault("responses", {}).setdefault(
                         status_code, {}
                     ).setdefault("content", {})["application/jsonl"] = jsonl_content
+                elif route.is_sse_stream:
+                    sse_content: dict[str, Any] = {}
+                    item_schema = copy.deepcopy(_SSE_EVENT_SCHEMA)
+                    if route.stream_item_field:
+                        content_schema = get_schema_from_model_field(
+                            field=route.stream_item_field,
+                            model_name_map=model_name_map,
+                            field_mapping=field_mapping,
+                            separate_input_output_schemas=separate_input_output_schemas,
+                        )
+                        item_schema["required"] = ["data"]
+                        item_schema["properties"]["data"] = {
+                            "type": "string",
+                            "contentMediaType": "application/json",
+                            "contentSchema": content_schema,
+                        }
+                    sse_content["itemSchema"] = item_schema
+                    operation.setdefault("responses", {}).setdefault(
+                        status_code, {}
+                    ).setdefault("content", {})["text/event-stream"] = sse_content
                 elif route_response_media_type:
                     response_schema = {"type": "string"}
                     if lenient_issubclass(current_response_class, JSONResponse):
```

### fastapi/routing.py::APIRoute.__init__
```diff
@@ -709,9 +807,16 @@ def __init__(
             else:
                 stream_item = get_stream_item_type(return_annotation)
                 if stream_item is not None:
-                    # Only extract item type for JSONL streaming when no
-                    # explicit response_class (e.g. StreamingResponse) was set
-                    if isinstance(response_class, DefaultPlaceholder):
+                    # Extract item type for JSONL or SSE streaming when
+                    # response_class is DefaultPlaceholder (JSONL) or
+                    # EventSourceResponse (SSE).
+                    # ServerSentEvent is excluded: it's a transport
+                    # wrapper, not a data model, so it shouldn't feed
+                    # into validation or OpenAPI schema generation.
+                    if (
+                        isinstance(response_class, DefaultPlaceholder)
+                        or lenient_issubclass(response_class, EventSourceResponse)
+                    ) and not lenient_issubclass(stream_item, ServerSentEvent):
                         self.stream_item_type = stream_item
                     response_model = None
                 else:
@@ -814,11 +919,16 @@ def __init__(
             name=self.unique_id,
             embed_body_fields=self._embed_body_fields,
         )
-        # Detect generator endpoints that should stream as JSONL
-        # (only when no explicit response_class like StreamingResponse is set)
-        self.is_json_stream = isinstance(response_class, DefaultPlaceholder) and (
+        # Detect generator endpoints that should stream as JSONL or SSE
+        is_generator = (
             self.dependant.is_async_gen_callable or self.dependant.is_gen_callable
         )
+        self.is_sse_stream = is_generator and lenient_issubclass(
+            response_class, EventSourceResponse
+        )
+        self.is_json_stream = is_generator and isinstance(
+            response_class, DefaultPlaceholder
+        )
         self.app = request_response(self.get_route_handler())
 
     def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
```

### fastapi/routing.py::get_request_handler
```diff
@@ -361,6 +368,7 @@ def get_request_handler(
         actual_response_class: type[Response] = response_class.value
     else:
         actual_response_class = response_class
+    is_sse_stream = lenient_issubclass(actual_response_class, EventSourceResponse)
     if isinstance(strict_content_type, DefaultPlaceholder):
         actual_strict_content_type: bool = strict_content_type.value
     else:
```

### fastapi/routing.py::get_request_handler.<locals>.app | fastapi/routing.py::get_request_handler.<locals>.app.<locals>._serialize_item
```diff
@@ -452,35 +460,125 @@ async def app(request: Request) -> Response:
         errors = solved_result.errors
         assert dependant.call  # For types
         if not errors:
-            if is_json_stream:
-                # Generator endpoint: stream as JSONL
+            # Shared serializer for stream items (JSONL and SSE).
+            # Validates against stream_item_field when set, then
+            # serializes to JSON bytes.
+            def _serialize_data(data: Any) -> bytes:
+                if stream_item_field:
+                    value, errors_ = stream_item_field.validate(
+                        data, {}, loc=("response",)
+                    )
+                    if errors_:
+                        ctx = endpoint_ctx or EndpointContext()
+                        raise ResponseValidationError(
+                            errors=errors_,
+                            body=data,
+                            endpoint_ctx=ctx,
+                        )
+                    return stream_item_field.serialize_json(
+                        value,
+                        include=response_model_include,
+                        exclude=response_model_exclude,
+                        by_alias=response_model_by_alias,
+                        exclude_unset=response_model_exclude_unset,
+                        exclude_defaults=response_model_exclude_defaults,
+                        exclude_none=response_model_exclude_none,
+                    )
+                else:
+                    data = jsonable_encoder(data)
+                    return json.dumps(data).encode("utf-8")
+
+            if is_sse_stream:
+                # Generator endpoint: stream as Server-Sent Events
                 gen = dependant.call(**solved_result.values)
 
-                def _serialize_item(item: Any) -> bytes:
-                    if stream_item_field:
```

## fastapi_5077
necessary groups: ['fastapi/dependencies/utils.py::get_typed_return_annotation', 'fastapi/dependencies/utils.py::get_typed_signature']

### fastapi/dependencies/utils.py::get_typed_return_annotation
```diff
@@ -217,12 +218,13 @@ def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
 
 def get_typed_return_annotation(call: Callable[..., Any]) -> Any:
     signature = inspect.signature(call)
+    unwrapped = inspect.unwrap(call)
     annotation = signature.return_annotation
 
     if annotation is inspect.Signature.empty:
         return None
 
-    globalns = getattr(call, "__globals__", {})
+    globalns = getattr(unwrapped, "__globals__", {})
     return get_typed_annotation(annotation, globalns)
 
 ```

### fastapi/dependencies/utils.py::get_typed_signature
```diff
@@ -192,7 +192,8 @@ def get_flat_params(dependant: Dependant) -> List[ModelField]:
 
 def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
     signature = inspect.signature(call)
-    globalns = getattr(call, "__globals__", {})
+    unwrapped = inspect.unwrap(call)
+    globalns = getattr(unwrapped, "__globals__", {})
     typed_params = [
         inspect.Parameter(
             name=param.name,
```

## requests_7505
necessary groups: ['src/requests/models.py::PreparedRequest.prepare_body', 'src/requests/models.py::RequestEncodingMixin._encode_files', 'src/requests/models.py::RequestEncodingMixin._encode_params']

### src/requests/models.py::PreparedRequest.prepare_body
```diff
@@ -641,7 +638,7 @@ def prepare_body(
             else:
                 if raw_data:
                     body = self._encode_params(raw_data)
-                    if isinstance(data, basestring) or isinstance(data, _SupportsRead):
+                    if isinstance(data, basestring) or _t.has_read(data):
                         content_type = None
                     else:
                         content_type = "application/x-www-form-urlencoded"```

### src/requests/models.py::RequestEncodingMixin._encode_files
```diff
@@ -236,9 +235,7 @@ def _encode_files(
 
             if isinstance(fp, (str, bytes, bytearray)):
                 fdata = fp
-            # data that proxies attributes to underlying objects needs hasattr
-            # defensive check for untyped callers
-            elif isinstance(fp, _SupportsRead) or hasattr(fp, "read"):
+            elif _t.has_read(fp):
                 fdata = fp.read()
             elif fp is None:  # defensive check for untyped callers
                 continue
```

### src/requests/models.py::RequestEncodingMixin._encode_params
```diff
@@ -161,7 +160,7 @@ def _encode_params(
 
         if isinstance(data, (str, bytes)):
             return data
-        elif isinstance(data, _SupportsRead):
+        elif _t.has_read(data):
             return data
         elif hasattr(data, "__iter__"):
             result: list[tuple[bytes, bytes]] = []
```

## rich_3782
necessary groups: ['rich/syntax.py::Syntax.__rich_console__', 'rich/syntax.py::Syntax._get_syntax']

### rich/syntax.py::Syntax.__rich_console__
```diff
@@ -626,7 +641,7 @@ def __rich_console__(
         self, console: Console, options: ConsoleOptions
     ) -> RenderResult:
         segments = Segments(self._get_syntax(console, options))
-        if self.padding:
+        if any(self.padding):
             yield Padding(segments, style=self._get_base_style(), pad=self.padding)
         else:
             yield segments
```

### rich/syntax.py::Syntax._get_syntax
```diff
@@ -640,15 +655,19 @@ def _get_syntax(
         Get the Segments for the Syntax object, excluding any vertical/horizontal padding
         """
         transparent_background = self._get_base_style().transparent_background
+        _pad_top, pad_right, _pad_bottom, pad_left = self.padding
+        horizontal_padding = pad_left + pad_right
         code_width = (
             (
                 (options.max_width - self._numbers_column_width - 1)
                 if self.line_numbers
                 else options.max_width
             )
+            - horizontal_padding
             if self.code_width is None
             else self.code_width
         )
+        code_width = max(0, code_width)
 
         ends_on_nl, processed_code = self._process_code(self.code)
         text = self.highlight(processed_code, self.line_range)```

## rich_4070
necessary groups: ['rich/_emoji_replace.py::_emoji_replace', 'rich/console.py::Console._caller_frame_info', 'rich/console.py::Console.export_html', 'rich/console.py::Console.export_svg', 'rich/console.py::Console.input', 'rich/console.py::Console.render', 'rich/protocol.py::rich_cast']

### rich/_emoji_replace.py::_emoji_replace
```diff
@@ -15,8 +12,10 @@ def _emoji_replace(
     _emoji_sub: _EmojiSubMethod = re.compile(r"(:(\S*?)(?:(?:\-)(emoji|text))?:)").sub,
 ) -> str:
     """Replace emoji code in text."""
+    from ._emoji_codes import EMOJI
+
     get_emoji = EMOJI.__getitem__
-    variants = {"text": "\uFE0E", "emoji": "\uFE0F"}
+    variants = {"text": "\ufe0e", "emoji": "\ufe0f"}
     get_variant = variants.get
     default_variant_code = variants.get(default_variant, "") if default_variant else ""
 
```

### rich/console.py::Console._caller_frame_info
```diff
@@ -1900,15 +1903,14 @@ def print_exception(
 
     @staticmethod
     def _caller_frame_info(
-        offset: int,
-        currentframe: Callable[[], Optional[FrameType]] = inspect.currentframe,
+        offset: int, currentframe: Optional[Callable[[], Optional[FrameType]]] = None
     ) -> Tuple[str, int, Dict[str, Any]]:
         """Get caller frame information.
 
         Args:
             offset (int): the caller offset within the current frame stack.
             currentframe (Callable[[], Optional[FrameType]], optional): the callable to use to
-                retrieve the current frame. Defaults to ``inspect.currentframe``.
+                retrieve the current frame. Defaults to None, which will use ``inspect.currentframe()``.
 
         Returns:
             Tuple[str, int, Dict[str, Any]]: A tuple containing the filename, the line number and
@@ -1920,17 +1922,22 @@ def _caller_frame_info(
         # Ignore the frame of this local helper
         offset += 1
 
-        frame = currentframe()
+        if currentframe is None:
+            import inspect
+
+            frame = inspect.currentframe()
+        else:
+            frame = currentframe()
         if frame is not None:
-            # Use the faster currentframe where implemented
             while offset and frame is not None:
                 frame = frame.f_back
                 offset -= 1
             assert frame is not None
             return frame.f_code.co_filename, frame.f_lineno, frame.f_locals
         else:
-            # Fallback to the slower stack
-            frame_info = inspect.stack()[offset]
+            from inspect import stack
```

### rich/console.py::Console.export_html
```diff
@@ -2242,6 +2253,8 @@ def export_html(
         Returns:
             str: String containing console contents as HTML.
         """
+        from html import escape
+
         assert (
             self.record
         ), "To export console contents set record=True in the constructor or instance"
```

### rich/console.py::Console.export_svg
```diff
@@ -2353,6 +2366,9 @@ def export_svg(
                 ids). If not set, this defaults to a computed value based on the recorded content.
         """
 
+        import zlib
+        from html import escape
+
         from rich.cells import cell_len
 
         style_cache: Dict[Style, str] = {}
```

### rich/console.py::Console.input
```diff
@@ -2166,7 +2175,9 @@ def input(
         if prompt:
             self.print(prompt, markup=markup, emoji=emoji, end="")
         if password:
-            result = getpass("", stream=stream)
+            import getpass as _getpass_mod
+
+            result = _getpass_mod.getpass("", stream=stream)
         else:
             if stream:
                 result = stream.readline()
```

### rich/console.py::Console.render
```diff
@@ -1321,7 +1314,7 @@ def render(
         render_iterable: RenderResult
 
         renderable = rich_cast(renderable)
-        if hasattr(renderable, "__rich_console__") and not isclass(renderable):
+        if hasattr(renderable, "__rich_console__") and not isinstance(renderable, type):
             render_iterable = renderable.__rich_console__(self, _options)
         elif isinstance(renderable, str):
             text_renderable = self.render_str(
```

### rich/protocol.py::rich_cast
```diff
@@ -28,7 +27,7 @@ def rich_cast(renderable: object) -> "RenderableType":
     from rich.console import RenderableType
 
     rich_visited_set: Set[type] = set()  # Prevent potential infinite loop
-    while hasattr(renderable, "__rich__") and not isclass(renderable):
+    while hasattr(renderable, "__rich__") and not isinstance(renderable, type):
         # Detect object which claim to have all the attributes
         if hasattr(renderable, _GIBBERISH):
             return repr(renderable)
```
